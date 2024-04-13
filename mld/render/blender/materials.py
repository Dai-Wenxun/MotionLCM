import bpy


def clear_material(material):
    if material.node_tree:
        material.node_tree.links.clear()
        material.node_tree.nodes.clear()


def colored_material_diffuse_BSDF(r, g, b, a=1, roughness=0.127451):
    materials = bpy.data.materials
    material = materials.new(name="body")
    material.use_nodes = True
    clear_material(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')
    diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
    diffuse.inputs["Color"].default_value = (r, g, b, a)
    diffuse.inputs["Roughness"].default_value = roughness
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    return material


# keys:
# ['Base Color', 'Subsurface', 'Subsurface Radius', 'Subsurface Color', 'Metallic', 'Specular', 'Specular Tint', 'Roughness', 'Anisotropic', 'Anisotropic Rotation', 'Sheen', 1Sheen Tint', 'Clearcoat', 'Clearcoat Roughness', 'IOR', 'Transmission', 'Transmission Roughness', 'Emission', 'Emission Strength', 'Alpha', 'Normal', 'Clearcoat Normal', 'Tangent']
DEFAULT_BSDF_SETTINGS = {"Subsurface": 0.15,
                         "Subsurface Radius": [1.1, 0.2, 0.1],
                         "Metallic": 0.3,
                         "Specular": 0.5,
                         "Specular Tint": 0.5,
                         "Roughness": 0.75,
                         "Anisotropic": 0.25,
                         "Anisotropic Rotation": 0.25,
                         "Sheen": 0.75,
                         "Sheen Tint": 0.5,
                         "Clearcoat": 0.5,
                         "Clearcoat Roughness": 0.5,
                         "IOR": 1.450,
                         "Transmission": 0.1,
                         "Transmission Roughness": 0.1,
                         "Emission": (0, 0, 0, 1),
                         "Emission Strength": 0.0,
                         "Alpha": 1.0}


def body_material(r, g, b, a=1, name="body", oldrender=True):
    if oldrender:
        material = colored_material_diffuse_BSDF(r, g, b, a=a)
    else:
        materials = bpy.data.materials
        material = materials.new(name=name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        diffuse = nodes["Principled BSDF"]
        inputs = diffuse.inputs

        settings = DEFAULT_BSDF_SETTINGS.copy()
        settings["Base Color"] = (r, g, b, a)
        settings["Subsurface Color"] = (r, g, b, a)
        settings["Subsurface"] = 0.0

        for setting, val in settings.items():
            inputs[setting].default_value = val

    return material


def floor_mat(color=(0.1, 0.1, 0.1, 1), roughness=0.127451):
    return colored_material_diffuse_BSDF(color[0], color[1], color[2], a=color[3], roughness=roughness)

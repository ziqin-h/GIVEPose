import bpy
from mathutils import Matrix, Vector

bpy.context.scene.render.filepath = './prepare_data/nocs_map_cube.png'
bpy.context.scene.render.alpha_mode = 'TRANSPARENT'  # in ['TRANSPARENT', 'SKY']
for item in bpy.data.objects:
    print(item)
    if item.type == 'MESH':

        vcol_layer = item.data.vertex_colors.new()

        for loop_index, loop in enumerate(item.data.loops):
            loop_vert_index = loop.vertex_index

            # here the scale is manually set for the cube to normalize it within [-0.5, 0.5]
            scale = 0.5

            # print("coord", scale*item.data.vertices[loop_vert_index].co)
            color = scale * item.data.vertices[loop_vert_index].co + Vector([0.5, 0.5, 0.5])
            # print(color)
            vcol_layer.data[loop_index].color = color

        item.data.vertex_colors.active = vcol_layer
        item.data.update()

        mat = bpy.data.materials.new('coord_color')

        mat.use_vertex_color_light = False
        mat.use_shadeless = True
        mat.use_face_texture = False
        mat.use_vertex_color_paint = True

        # if item.data.materials:
        #    for i in range(len(item.data.materials)):
        #        item.data.materials[i] = mat
        #    else:
        item.data.materials.clear()
        item.data.materials.append(mat)
        item.active_material = mat

bpy.ops.render.render(write_still=True)
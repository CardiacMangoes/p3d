import pickle, bpy, os, numpy as np 
from mathutils import Vector

def generate_object(params, clear=1): 
    print('STARTING')     
    if clear: delete_everything() 
    # generate object from params
    bpy.ops.mesh.shape_generator(
    random_seed=params['seed'],
    mirror_x=0,
    mirror_y=0,
    mirror_z=0,
    favour_vec=params['axis_preferences'],
    amount=params['n_extrusions'],
    is_subsurf=(params['n_subdivisions']>0),
    subsurf_subdivisions=params['n_subdivisions'],
    big_shape_num=params['n_bigshapes'],
    medium_shape_num=params['n_medshapes'],
    medium_random_scatter_seed=params['med_locseed'],
    big_random_scatter_seed=params['big_locseed'],
    big_random_seed=params['big_shapeseed'],
    medium_random_seed=params['med_shapeseed'], 
#    medium_shape_scale=params['medium_shape_scale'], 
    )
    print('OVER') 
    # unselect everything 
    [i.select_set(False) for i in bpy.data.objects]

def scale_and_center_object(xyz=(0,0,0)):
    """
    thanks to:
    https://blender.stackexchange.com/questions/137485/how-to-scale-primitive-shapes-in-blender-so-that-they-are-within-a-unit-sphere
    """
    def bbox(ob):
        return (Vector(b) for b in ob.bound_box)
    def bbox_center(ob):
        return sum(bbox(ob), Vector()) / 8
    def mesh_radius(ob):
        o = bbox_center(ob)
        return max((v.co - o).length for v in ob.data.vertices)

    object = bpy.context.selected_objects[0]
    scale_factor = 1 / mesh_radius(object)
    object.scale *= scale_factor
    obj = bpy.context.active_object
    obj.location = (xyz[0],xyz[1],xyz[2])
    
def delete_everything(): 
    [bpy.data.collections.remove(i) for i in bpy.data.collections]
    [bpy.data.objects.remove(i) for i in bpy.data.objects]

if __name__ == '__main__': 
    print('\n\n\n\n-------LOADING SHAPE GENERATOR------\n\n\n\n')
    base_dir = os.getcwd()
    object_dir = os.path.join(base_dir, 'objects')

    if not os.path.exists(object_dir):
        os.makedirs(object_dir)
    shapegen_dir = os.path.join(base_dir, 'shape_generator')
    bpy.ops.preferences.addon_install(filepath=shapegen_dir)
    bpy.ops.preferences.addon_enable(module='shape_generator') 
    bpy.ops.wm.save_userpref()
    print('\n\n\n-------FINISHED LOADING SHAPE GENERATOR\n\n\n\n')  
    random_seed = np.random.randint(1000)
    np.random.seed(random_seed) 

    params = {'axis_preferences':[.2, .8, 1],
        'n_bigshapes': 2,
        'n_medshapes': 1,
        'big_shapeseed':8, 
        'big_locseed': 0,
        'med_locseed': 3, 
        'med_shapeseed': 7, # 13],
        'n_extrusions': 4,
        'seed':0,
        'subdivisions': [0, 5], 
        'brotations': [0, .2], 
        'mrotations': [0, .2], 
#        'medium_shape_scale': .7, 
        }

#    params = {
#        'axis_preferences':[np.random.uniform(0, .2), np.random.uniform(.6, 1), np.random.uniform(.6, 1)],
#        'n_bigshapes': 2,
#        'n_medshapes': 1,
#        'big_shapeseed': np.random.randint(1000), 
#        'big_locseed': 0,
#        'med_locseed': np.random.randint(1000), # 3, 9 
#        'med_shapeseed': np.random.randint(1000),
#        'n_extrusions': np.random.randint(3, 7),
#        'seed':random_seed,
#        'subdivisions': [1], 
#        'brotations': [np.random.uniform(-.5, .5), np.random.uniform(-.5, .5)], 
#        'mrotations': [np.random.uniform(-.5, .5), np.random.uniform(-.5, .5)]}

    print( params ) 

    collections = bpy.data.collections        
    object_jitter = .1
    stimulus_info = {} 
    this_folder = os.path.split(os.getcwd())[1]
    object_folder = os.path.join( os.getcwd(), 'objects') # '/Users/biota/working/perirhinal_function/stimuli/objects'
    save_location = object_folder #os.path.join( object_folder, this_folder)
    save_string = '%003d_%d%d%d'
    save_objects = 1 
    
    for i_subdivisions in range(len(params['subdivisions'])): 
        
        params['n_subdivisions'] = params['subdivisions'][i_subdivisions]
        
        for big_twist in range(len(params['brotations'])): 
            
            for med_twist in range(len(params['mrotations'])):         

                generate_object(params)
            
                med_coll = [i.name for i in collections if 'Medium' in i.name]
                med_objects = [i.name for i in collections[med_coll[0]].objects]
                
                big_coll = [i.name for i in collections if 'Generated Shape Collection'== i.name]
                big_objects = [i.name for i in collections[big_coll[0]].objects]

                for i_object in [i for i in big_objects if i != 'Generated Shape']: 
                    
                    i_rotation = params['brotations'][big_twist]*np.pi
                    bpy.data.objects[i_object].rotation_euler.x -= i_rotation
                    
                for i_object in med_objects: 
                    
                    i_rotation = params['mrotations'][med_twist]*np.pi
                    bpy.data.objects[i_object].rotation_euler.x -= i_rotation
                    #bpy.data.objects[i_object].rotation_euler.y -= np.random.uniform(-.2,.2)
                    #ymed, zmed = np.random.uniform(-.2,.2), np.random.uniform(-.1,.1)
                    #bpy.data.objects[i_object].location += Vector( [0, ymed, zmed ]  )
                    bpy.data.objects[i_object].location.x += .3#np.random.uniform(.2,.4)

                [i.select_set(True) for i in bpy.data.objects]
                
                bpy.ops.object.join()
                bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
                bpy.ops.object.location_clear(clear_delta=False)
                # save a single object scaled and located at the origin
                scale_and_center_object()
                object_name = save_string%(params['big_shapeseed'], i_subdivisions, big_twist, med_twist )
                save_name = os.path.join(save_location,  object_name + '.obj')
                if save_objects: bpy.ops.export_scene.obj(filepath=save_name, use_selection=True)
    
    
    stimulus_info['%003d'%params['big_shapeseed']] = params
    metadata_filename = os.path.join(save_location, 'metadata.pickle')
    print('\n\n\n--------------------------RUNNING----------------')
    with open(metadata_filename, 'wb') as handle:
        pickle.dump(stimulus_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print( 'metadata saved:', stimulus_info)

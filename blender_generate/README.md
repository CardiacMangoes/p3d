# Running Shapegen
## Installing Blender

1. Download and untar Blender
```
wget https://mirror.clarkson.edu/blender/release/Blender2.90/blender-2.90.1-linux64.tar.xz
tar -xvf blender-2.82a-linux64.tar.xz 
```

2. Install other Python dependencies in the Blender bundled Python
```
cd $INSTALL_PATH/blender-2.82a-linux64/2.82/python/bin/
./python3.7m -m ensurepip
./pip3 install numpy scipy
```

3. In your `.bash_aliases` file, add
```
alias bpy="blender --background -noaudio --python”
```
and run
```
source ~/.bashrc
```

install blender, from https://github.com/sxyu/pixel-nerf/tree/master/scripts
`` wget https://mirror.clarkson.edu/blender/release/Blender2.90/blender-2.90.1-linux64.tar.xz --no-check-certificate``


## Make Objects
1. Navigate to `blender_generate`
 ```
cd blender_generate
```
2. Run
```
bpy generate_objects.py
```

## Render Images

1. Run
```
bpy generate_images.py
```

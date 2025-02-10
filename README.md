# P3d
Perceptive 3D (interim clean title)

## Setting up Conda Enviroment

Create a conda environment

```bash
conda create --name p3d -y python=3.10
conda activate p3d
python -m pip install --upgrade pip
```

In this directory run
```bash
pip install -e .
```

## Render Viewpoints

```bash
python p3d/scripts/viser_render.py [file path to obj]
```

Viewpoint Comparison: `analysis.ipynb`
Generative Comparison: `test_models.ipynb`

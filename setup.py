from setuptools import setup, find_packages

setup(
    name='dust3r',
    version='1.0.0',
	packages=find_packages(),
	install_requires=[
		'torch',
		'torchvision',
		'tqdm',
		'opencv-python',
		'pillow',
		'scipy',
		'huggingface_hub',
		'einops',
		'croco@git+https://github.com/Linusnie/croco.git',
	]
)
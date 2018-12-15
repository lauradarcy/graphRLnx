import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='graphRLnx',
	version='0.0.2',
	author='Laura D\'Arcy',
	author_email='luludarcy@gmail.com',
	description='a directed acyclic graph environment for use with openAI gym, with a networkX backend',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/luludarcy/graphRLnx',
	packages=setuptools.find_packages(),
	install_requires=['gym', 'networkx'],  # And any other dependencies graphRL needs
	classifiers=[
		"Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

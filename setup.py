
"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="py2vision", # Replace with your own username
    version="1.0.3",
    license='MIT',
    author="Guillermo Jose Raven Lusinche",
    author_email="guillermoraven96@gmail.com",
    description="A package to implement a stereo vision system trained with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/corvus96/PyTwoVision",
    packages=setuptools.find_packages(),
    package_dir={'py2vision': 'py2vision'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires='>=3.6',
     # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'numpy == 1.21.5',
        'tensorflow == 2.8.0',
        'opencv-contrib-python==4.6.0.66',
        'wget == 3.2',
        'pandas',
        'pyyaml', 
        'h5py'
    ]
)
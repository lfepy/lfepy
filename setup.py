from setuptools import setup, find_packages

setup(
    name='lfepy',
    version='1.0.7',
    author='Dr. Prof. Khalid M. Hosny, BSc. Mahmoud A. Mohamed, Dr. Rania Salama, Dr. Ahmed M. Elshewey',
    author_email='lfepy@gmail.com',
    description='lfepy is a Python package for local feature extraction.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/lfepy/lfepy',  # Replace with your repository URL
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # List your library's dependencies here
        'numpy>=1.26.4',
        'scipy>=1.13.0',
        'scikit-image>=0.23.2',
    ],
    test_suite='tests',
    tests_require=[
        'unit test',  # Example test framework
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Update according to your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.0',
)
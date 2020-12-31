from setuptools import find_packages, setup

setup(
    name='cassava',
    version='0.0.1',
    url='https://github.com/tety94/cassava-leaf-disease',
    license='',
    author='stefano',
    author_email='callegarostefano@gmail.com',
    description='Cassave Leaf, Kaggle competition',
    scripts=[
        "classes/json_analizer.py",
        "classes/pandas_utils.py",
    ],
    include_package_data=True,
    packages=find_packages(),
    zip_safe=False,
)

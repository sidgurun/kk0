import setuptools

from setuptools.command.develop import develop
from setuptools.command.install import install

import LyaRT_Grid as LG

#====================================================================#

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        LG.Download_data()

        develop.run(self)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        LG.Download_data()

        install.run(self)

#====================================================================#


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LyaRT_Grid",
    version="0.1.4",
    author="Siddhartha Gurung Lopez",
    author_email="sidgurung@cefca.es",
    description="Fast Lyman alpha Radiative Transfer for everyone!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sidgurun/LyaRT-Grid",
    packages=setuptools.find_packages(),
    #install_requires=['scikit-learn'],
    install_requires=setuptools.find_packages(),
    include_package_data = True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    cmdclass={ 'develop': PostDevelopCommand,
               'install': PostInstallCommand, },
    #test_suite='nose.collector',
    #tests_require=['nose'],
)


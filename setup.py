import versioneer

from setuptools import setup, find_packages
requirements = [
    "numba",
    "numpy>=1.20.3",
    "estraces>=1.9.4",
    "psutil",
    "scipy"
]


setup(
    name='scared',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="eshard scared Python library.",
    author="eshard",
    license="Proprietary",
    author_email='contact@eshard.com',
    packages=find_packages(include=["scared", "scared.*"]),
    install_requires=requirements,
    keywords='scared',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Topic :: Software Development',
    ]
)

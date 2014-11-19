from setuptools import setup, find_packages

setup(
	name='foxhound', 
	version='0.1',
    packages=find_packages(),
    description="""
        Scikit learn inspired library for gpu-accelerated machine learning
    """,
    license="MIT License (See LICENSE)",
    url="https://github.com/IndicoDataSolutions/indicoio-python",
    author="Alec Radford, Madison May",
    author_email="""
        Alec Radford <madison@indico.io>,
        Madison May <madison@indico.io>
    """,
    install_requires=[
        "six >= 1.3.0",
        "numpy >= 1.8.1",
        "scipy >= 0.14.0",
        "Theano >= 0.6.0",
    ],
)



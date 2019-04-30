import io

from setuptools import find_packages, setup

with io.open('README.md', 'rt', encoding='utf8') as f:
    readme = f.read()

description = "Openstax response validator server"

setup(
    name='response-validator',
    version='1.0.0',
    url='https://github.com/openstax/response-validator',
    license='AGPL, See also LICENSE.txt',
    Author='Openstax Team',
    maintainer_email='info@cnx.org',
    description=description,
    long_description=readme,
    packages=find_packages(),
    package_data={"validator": ["ml/data/*csv", "ml/corpora/*.txt"]},
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'flask',
    ],
    extras_require={
        'test': [
            'pytest',
            'coverage',
        ],
    },
)

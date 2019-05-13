import io

import nltk

import versioneer

from setuptools import find_packages, setup

with io.open('README.md', 'rt', encoding='utf8') as f:
    readme = f.read()

for data_file in ('stopwords', 'words', 'punkt'):
    nltk.download(data_file, download_dir='validator/ml/corpora/nltk_data')

description = "Openstax response validator server"

setup(
    name='response-validator',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url='https://github.com/openstax/response-validator',
    license='AGPL, See also LICENSE.txt',
    Author='Openstax Team',
    maintainer_email='info@cnx.org',
    description=description,
    long_description_content_type='text/markdown',
    long_description=readme,
    packages=find_packages(),
    package_data={"validator": ["ml/data/*csv",
                                "ml/corpora/*.txt",
                                "ml/corpora/nltk_data/*/*",
                                "ml/corpora/nltk_data/*/*/*",
                                "ml/corpora/nltk_data/*/*/*/*",
                                ]},
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'flask>=1.0.2',
        'flask-cors',
        'pandas',
        'nltk',
        'symspellpy',
    ],
    extras_require={
        'test': [
            'pytest',
            'coverage',
        ],
    },
)

from setuptools import setup

setup(
    name='tfidfmatcher',
    version='0.0.2',
    description=(
        'Fast partial string match using cosine similarity on tf-idf vectors'
    ),
    url='https://github.com/adamhajari/tfidf',
    author='Adam Hajari',
    author_email='adamhajari@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
    ],
    keywords='partial string matching tfidf',
    include_package_data=True,
    packages=['tfidfmatcher'],
    install_requires=[
        "scikit-learn>=0.19.1",
        "numpy>=1.14.3"
    ]
)

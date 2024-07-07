from setuptools import setup, find_packages

setup(
    name='MultiExtractiveSummarizer',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        "nltk",
        "sentence-transformers",
        "scipy",
        "scikit-learn"
        
    ],
    author='Arshraj Randhawa',
    author_email='arshraj.randhawa@gmail.com',
    description='Python package for extractive text summarization using various embeddings and methods.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/arshraj-r/MultiExtractiveSummarizer.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

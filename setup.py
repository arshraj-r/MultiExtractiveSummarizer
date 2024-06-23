from setuptools import setup, find_packages

setup(
    name='MultiExtractiveSummarizer',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'sentence-transformers',
        'torch',
        'numpy>=1.14,<2',
        'scikit-learn',
        'gensim',
        'sumy'
    ],
    entry_points={
        'console_scripts': [
            'ExtractiveSummarizer=ExtractiveSummarizer.summarizer:ExtractiveSummarizer',
        ],
    },
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

from setuptools import setup, find_packages

setup(
    name='text_summarizer',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'sentence-transformers',
        'torch',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'summarize=text_summarizer.summarizer:ExtractiveSummarizer',
        ],
    },
    author='Your Name',
    author_email='your_email@example.com',
    description='An extractive text summarizer using Sentence-BERT',
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

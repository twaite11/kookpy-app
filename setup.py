from setuptools import setup, find_packages

setup(
    name='kookpy',
    version='1.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'tensorflow',
        'joblib',
        'streamlit',
        'requests',
        'scikit-learn',
        'plotly',
        'numpy',
        'bcrypt',
    ],
)
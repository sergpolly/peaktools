from setuptools import setup, find_packages

setup(
    name='peaktools',
    version='0.0.1',
    # py_modules=['peaktools'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=['Click','pandas','numpy'],
    entry_points={
        'console_scripts': [
        'peaktools = peaktools:cli',
        ]
    }
)

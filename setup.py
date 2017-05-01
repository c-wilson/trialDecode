# from distutils.core import setup
from setuptools import setup

setup(
    name='trialDecode',
    version='0.1.0',
    packages=['trialDecode'],
    url='',
    license='MIT',
    author='chris',
    author_email='cdw291@nyu.edu',
    description='', install_requires=[],
    entry_points={
        'console_scripts': ['trialDecode = trialDecode.__main__:main']
    }
)

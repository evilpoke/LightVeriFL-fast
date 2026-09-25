from setuptools import setup, find_packages
import sys

try:
    from setuptools import setup, find_packages
except ImportError:
    print("Please install or upgrade setuptools or pip to continue")
    sys.exit(1)

setup(
    name='lightverirepo',
    version='1.0',
    description='Python tool for lightverifl with lightsegacc',
    keywords='autonomous automated vessels motion planning',
    author='Stefan Schaerdinger',
    author_email='stefan.schaerdinger@tum.de',
    license="BSD",
    packages=find_packages(exclude=['build']),

)
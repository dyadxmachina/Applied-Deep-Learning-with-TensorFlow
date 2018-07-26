from setuptools import setup

REQUIRED_PACKAGES = [
    'pandas >= 0.22.0'
]
if __name__ == '__main__':
    setup(name='train',
          packages=['train'],
          install_requires=['tf-nightly-gpu'])
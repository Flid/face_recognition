from setuptools import setup, find_packages

setup(
    name='face_recognition',
    version=__import__('face_recognition').__version__,
    description='Set of libraries to analyze camera stream, detect and recognize faces',
    author='Anton Kirilenko',
    url='https://github.com/Flid/face_recognition',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
    ],
    include_package_data=True,
)

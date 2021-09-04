#updating
# echo 'Updating System'
# sudo apt-get update

# # #install python
# echo 'Installing Python'
# wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
# sha256sum Miniconda3-py39_4.9.2-Linux-x86_64.sh
# bash Miniconda3-py39_4.9.2-Linux-x86_64.sh 
# source "$HOME/miniconda/etc/profile.d/conda.sh"
# conda config --set auto_activate_base false
# source ~/.bashrc

# #environment
# echo "Creating Environment"
# conda create --name prrana python=3.8
# conda activate prrana

#install RDkit
echo 'Installing RDKIT'
conda install -n prrana -c conda-forge rdkit

#numpy
echo 'Installing NUMPY'
conda install -n prrana numpy

#scipy
echo 'Installing SCIPY'
conda install -n prrana scipy

#matplotlib
echo 'Installing MATPLOTLIB'
pip install matplotlib

#pandas
echo 'Installing PANDAS'
conda install -n prrana pandas

#regex
echo 'Installing REGEX'
conda install -n prrana regex

#statsModel
echo 'Installing STATMODELS'
conda install -n prrana -c conda-forge statsmodels

#collection
echo 'Installing collections-libs'
pip install collections-extended

#random
echo 'Installing RANDOM'
pip install random2

#seaborn
echo 'Installing SEABORN'
pip install seaborn 

#sklearn
echo 'Installing SKLEARN'
conda install -n prrana -c conda-forge scikit-learn

# #coverage
# echo 'Installing COVERALL'
# conda install -n prrana -c conda-forge coveralls

# #codecov
# echo 'Installing CODECOVERAGE'
# conda install -n prrana -c conda-forge codecov

#latex
echo 'Installing texlive'
sudo apt-get install -y texlive texlive-latex-recommended
sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super

#double check
source ~/.bashrc
conda activate prrana
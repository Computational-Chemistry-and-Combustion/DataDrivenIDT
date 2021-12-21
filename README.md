[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5774617.svg)](https://doi.org/10.5281/zenodo.5774617)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5795476.svg)](https://doi.org/10.5281/zenodo.5795476)
[![CodeFactor](https://www.codefactor.io/repository/github/computational-chemistry-and-combustion/datadrivenidt/badge/main)](https://www.codefactor.io/repository/github/computational-chemistry-and-combustion/datadrivenidt/overview/main)
[![codecov](https://codecov.io/gh/Computational-Chemistry-and-Combustion/DataDrivenIDT/branch/main/graph/badge.svg?token=N83D9RS9HQ)](https://codecov.io/gh/Computational-Chemistry-and-Combustion/DataDrivenIDT)
[![Build Status](https://app.travis-ci.com/Computational-Chemistry-and-Combustion/DataDrivenIDT.svg?branch=main)](https://app.travis-ci.com/Computational-Chemistry-and-Combustion/DataDrivenIDT)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FComputational-Chemistry-and-Combustion%2FData_driven_Kinetics.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2FComputational-Chemistry-and-Combustion%2FData_driven_Kinetics?ref=badge_shield)
![GitHub top language](https://img.shields.io/github/languages/top/Computational-Chemistry-and-Combustion/DataDrivenIDT)
![GitHub](https://img.shields.io/github/license/Computational-Chemistry-and-Combustion/DataDrivenIDT)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/Computational-Chemistry-and-Combustion/DataDrivenIDT)
[![Open Source Love](https://img.shields.io/badge/Open-source-%3C3)](https://img.shields.io/badge/Open-source-%3C3)
[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2FComputational-Chemistry-and-Combustion%2FDataDrivenIDT)](https://twitter.com/intent/tweet?text=Want\+to\+predict\+ignition\+delay\+?\+Try\+out\+DataDrivenIDT\+Framework:&url=https%3A%2F%2Fgithub.com%2FComputational-Chemistry-and-Combustion%2FDataDrivenIDT)


## About The Project:

:fire:  This repository can be used to develop training models mentioned and make prediction using those models. The frame-work is designed for the ignition delay  data but with minor changes it works perfectly fine with any data having continuous dependent (output) variable.  Please look at the manual for more information.

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#system-Requirements">System Requirements<a>
    </li>
    <li>
	<a href="#installation">Installation</a>
      <ul>
        <li><a href="#add-sourcing-to-find-command">Add sourcing to find command<a></li>
        <li><a href="#Install-dependency">Install dependency<a></li>
        <li><a href="#make-run.sh-file-executable">Make Run.sh file executable<a></li>
      </ul>
    </li>
    <li><a href="#commands-to-run-the-program">Commands to run the program</a></li>
    <li><a href="#examples">Examples</a></li>
    <ul>
        <li>    <a href="#example-1">Example:1<a></li>
        <li>    <a href="#example-2">Example:2<a></li>
      </ul>
    <li><a href="#brought-up-by">Brought up by</a></li>
  </ol>
</details>

---
## System Requirements:

:fire:  OS : Linux

:fire:  Python 3.6+

---
## Installation:

:fire:  Clone the repository in suitable directory.

:fire:  Open your **./bashrc** file and add lines given below at the bottom of file.


---
**Add sourcing to find command:**

:fire:  Copy the following commands in your **./bashrc** file 

```sh
##Package command Finder:
export IDCODE="${HOME}/PathToDir/.../Data_driven_Kinetics/"
export PATH=$PATH:$IDCODE
alias IDprediction="pwd>${HOME}/PathToDir/.../Data_driven_Kinetics/filelocation.txt && Run.sh"
```
Replace "/PathToDir/.../" with your directory location.

--- 
**Example:**

If repo is cloned in **./home** directory then configure **.bashrc** using following command:


```sh
##Package command Finder:
export IDCODE="${HOME}/Data_driven_Kinetics/"
export PATH=$PATH:$IDCODE
alias IDprediction="pwd>${HOME}/Data_driven_Kinetics/filelocation.txt && Run.sh"
```
---
**Source the changes:**

:fire:  **(IMPORTANT)** To configure the changes in .bashrc, write following command in terminal.

```sh
cd
source .bashrc
```
---
**Install dependency:**

:fire:  To install all the dependency use INSTALL.sh file. Write the commands given below in the terminal

```sh
chmod +x INSTALL.sh

./INSTALL.sh
```
 
 ---
**Make Run.sh file executable:**

:fire:  To make run file executable, go to **./Data_driven_Kinetics** and write following command.

```sh
chmod +x Run.sh
```
 

---

## Commands to run the program:

All set!

Now, open terminal and type following commands to generate result.

```sh
IDprediction --flag argument
```

Input arguments to 'IDprediction' are specified as below:

Consider the data file as 'file_name.csv'


:fire:  **-\-a** : ‘**A**nalyze’ the data-set by selecting certain parameters

```sh
IDprediction --a  manual --sfile file_name.csv 
```

:fire:  **-\-b** : Find types of '**b**ond’ associated with given fuel
```sh
IDprediction --b  FuelSMILES
IDprediction --b CCC
IDprediction --b CCCCCC

```

:fire:  **-\-h** : Generates '**h**istogram’ plots of parameters for each fuel individually

```sh
IDprediction --h  file_name.csv 
```
:fire:  **-\-train**  : To train the algorithm using passed data

:fire:  **-\-test**  : To test the algorithm using passed data

:fire:  **-\-c**  : To define the '**c**riteria' for error based clustering

:fire:  **-\-l**   : To ‘**l**imit’ number of reference point

:fire:  **-\-r**   : To '**r**emove’ feature by back-elimination

:fire:  **-\-s**  : To specify **s**ignificance level

:fire:  **-\-n**  : To specify **n**umber of clusters

:fire:  **-\-i**  : To specify **I**terations limit

:fire:  **-\-algo GMM** : To train the model using Gaussian Mixture Modelling

```sh
IDprediction --algo GMM --train trainset.csv --n 3
IDprediction --algo GMM --test testset.csv
```
**Don’t forget to add  ’feature selection.py
file’ for non-fuel data**

:fire:  other implemented algorithms

```sh
IDprediction --algo tree --train trainset.csv --c 0.1 --l 20
IDprediction --algo tree --test testset.csv

IDprediction --algo multi --train trainset.csv --r True --s 0.1
IDprediction --algo multi --test testset.csv

IDprediction --algo spath --train trainset.csv --i 200 --n 3
IDprediction --algo spath --test testset.csv
```

:fire:  **--k**  : To run code multiple ‘(**k**)’ times and store all test prediction result in different directory

```sh
IDprediction --k testset.csv
```

:fire:  **--f**  : Probability density ‘**f**unction’ plot of testing result after running code 'k' times

```sh
IDprediction --f testset.csv
```


:fire:  **--p**  : **P**lot and obtain of average value of coefficient from coefficient file (If coefficient result obtained many times and there is variation in coefficients)
```sh
IDprediction --p  coefficient_3.csv 
```

---
## Examples:

**Example:1**
Run the following commands to generate models and make predictions using Ignition delay data:
```sh
cd TryYourself/nAlkaneIDT/
IDprediction --algo GMM --train trainset.csv --n 3
IDprediction --algo GMM --test testset.csv
```

**Example:2**
Run the following commands to generate models and make predictions using Wine quality data:
```sh
cd TryYourself/WineQuality/
IDprediction --algo GMM --train trainset.csv --n 5
IDprediction --algo GMM --test testset.csv
```

Make appropriate changes in **’feature selection.py'** file to change features accordingly to the data. (Check manual)

---

## Brought up by:

<dl>
      <a href="https://krithikasivaram.github.io">
         <img alt="CCC Group" src="https://github.com/pragneshrana/logos/blob/master/c3/image.png"
         width=100" height="100">
      </a>
      <a href="http://sivaramambikasaran.com/">
         <img alt="SAFRAN Group" src="https://github.com/pragneshrana/logos/blob/master/safran/17197871.png"
         width=100" height="100">
      </a>
</dl>

## License
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FComputational-Chemistry-and-Combustion%2FDataDrivenIDT.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2FComputational-Chemistry-and-Combustion%2FDataDrivenIDT?ref=badge_large)

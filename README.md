

<a href="https://zenodo.org/badge/latestdoi/373245755"><img src="https://zenodo.org/badge/373245755.svg" alt="DOI"></a>



## About The Project:

:fire:  This repository can be used to develop training models using tree type regression-based clustering and make prediction using those models. The frame-work is customized for the ignition delay  data but with minor changes it works fine with any data having continuous dependent (output) variable. The algorithm uses error-based technique to divide the data into three clusters based on relative error in prediction and sign of prediction error to obtain the accurate regression models. Please look at the manual for more information.

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
IDprediction -flag file_name.csv
```

Input arguments to 'IDprediction' are specified as below:

Consider the data file as 'file_name.csv'


:fire:  **-a** : ‘**A**nalyze’ the data-set by selecting certain parameters

```sh
IDprediction -a  file_name.csv  
```

:fire:  **-b** : Find types of '**b**ond’ associated with given fuel
```sh
IDprediction -b  FuelSMILES
IDprediction -b CCC
IDprediction -b CCCCCC

```

:fire:  **-h** : Generates '**h**istogram’ plots of parameters for each fuel individually

```sh
IDprediction -h  file_name.csv 
```

:fire:  **-c**  : To define the '**c**riteria' for error based clustering

:fire:  **-l**   : To ‘**l**imit’ number of reference point

:fire:  **-r**   : To '**r**emove’ feature by back-elimination

:fire:  **-s**  : To specify **s**ignificance level


:fire:  **-m** : To find out **m**ultiple linear regression of data 

```sh
IDprediction -c 0.05 -l 10 -r True -s 0.05  -m  file_name.csv 
```

:fire: **-t**  : ‘**T**ree’ type regression based clustering algorithm

```sh
IDprediction -c 0.05 -r False -t file_name.csv 
```

:fire:  **-e** : '**E**xternal' Dataset used for prediction (Complete above Model generation first)

```sh
IDprediction -e  test_data.csv 
```

:fire:  **-k**  : To run code multiple ‘(**k**)’ times and store all test prediction result in different directory

```sh
IDprediction -k testset.csv
```

:fire:  **-f**  : Probability density ‘**f**unction’ plot of testing result after running code 'k' times

```sh
IDprediction -f testset.csv
```


:fire:  **-p**  : **P**lot and obtain of average value of coefficient from coefficient file (If coefficient result obtained many times and there is variation in coefficients)
```sh
IDprediction -p  coefficient_3.csv 
```

:fire:  **-o**  : To run any '**o**ther’ dataset than fuel

```sh
IDprediction -c 0.05 -l 10 -o anyFile.csv
```
**Don’t forget to make changes in ’feature selection.py
file’**

---
## Examples:

**Example:1**
Run the following commands to generate models and make predictions using Ignition delay data:
```sh
cd TryYourself/nAlkaneIDT/
IDprediction -c 0.1 -t trainset.csv
IDprediction -e testset.csv
```

**Example:2**
Run the following commands to generate models and make predictions using Wine quality data:
```sh
cd TryYourself/WineQuality/
IDprediction -c 0.1 -o trainset.csv
IDprediction -e testset.csv
```

Make appropriate changes in **’feature selection.py'** file to change features accordingly to the data. (Check manual)

---

## Brought up by:

<dl>
      <a href="https://krithikasivaram.github.io">
         <img alt="CCC Group" src="https://github.com/pragneshrana/logos/blob/master/logo.jpg"
         width=100" height="100">
      </a>
      <a href="http://sivaramambikasaran.com/">
         <img alt="SAFRAN Group" src="https://github.com/pragneshrana/logos/blob/master/17197871.png"
         width=100" height="100">
      </a>
</dl>


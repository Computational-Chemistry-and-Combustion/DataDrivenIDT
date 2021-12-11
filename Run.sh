#!/bin/bash -e

echo
echo 'WELCOME TO THE PREDICTIVE WORLD '
echo 
echo

cd $HOME
code_directory=$IDCODE
# echo "Current path is : $code_directory"

#chnaging and storing path
cd $code_directory
curr_location=$(<$code_directory/filelocation.txt) 
cd ./src

## IT will check if you are running on travis CI then you 
## Coverage run to get generate report else python3
string=$IDCODE
if [[ $string == *"travis"* ]]; 
then
    runVar='coverage run'
else
    runVar='python3'
fi

error_criteria=0.05 #Tree:erroe based clustering criteria
significance_level=0.05 #significance level
max_iter=100
elimination="False" #elimination default set as Flase
limited_ref_points="False"
method="common"
process="general"
num_clusters=2
##################################################
## List of arguments in the 
##################################################
ARGUMENT_LIST=(
    "algo"
    "train"
    "test"
    "trainO"
    "testO"
    "sfile"
    "c"
    "b"
    "a"
    "i"
    "h"
    "m"
    "e"
    "k"
    "f"
    "n"
    "p"
    "r"
    "s"
    "d"
    "o"
    "l"
)


# read arguments
opts=$(getopt \
    --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "" \
    -- "$@"
)

eval set --$opts


#$# will number of arguments passed
#run till all arnuments passed
num=$(( ($#-1)/2 ))  #defined variable to stoe number of flags passed

# if not argument is passed then show them commands
if(($num == 0))
then 
echo ""
echo "No argument passed : "
echo " --algo : Select algorithm Multiple regression, Spath, GMM, Error based tree"
echo " --train : Select algorithm Multiple regression, Spath, GMM, Error based tree"
echo " --test : Select algorithm Multiple regression, Spath, GMM, Error based tree"
echo " --trainO : Select algorithm Multiple regression, Spath, GMM, Error based tree"
echo " --testO : Select algorithm Multiple regression, Spath, GMM, Error based tree"
echo " --sfile : Select algorithm Multiple regression, Spath, GMM, Error based tree"
echo " --c : Critria for error based clustering"
echo " --l : Limit of number of refrence point"
echo " --b : Bond analysis in the Fuel "
echo " --a : Analysis for fuel parameters"
echo " --i : Maximum number of iteration"
echo " --n : Maximum number of cluster"
echo " --h : Histogram Plots"
echo " --f : Plot histogram of all the result"
echo " --k : External fuel analysis saving all the result"
echo " --p : plotting of coefficient to find Average values"
echo " --r : Backward elimination Activation True/False"
echo " --s : Significance Level for Backward Elimination"
echo " --d : Dataset generation ONLY for fuel contaning SMILES"
echo ""
exit
fi

while [ $num -gt 0 ]; do
    case "$1" in
    # define methods over here
    "--algo")
        method=$2 #which algorithm to execute
        echo "Reading the  algorithm :"$method
        shift 2
        ;;
    "--train")
        dataset_location="$curr_location/"$2
        echo "Reading the  data location :"$dataset_location
        process="train"
        shift 2
        ;;
    "--test")
        dataset_location="$curr_location/"$2
        echo "Reading the  data location :"$dataset_location
        process="test"
        shift 2
        ;;
    "--trainO")
        dataset_location="$curr_location/"$2
        echo "Reading the  data location :"$dataset_location
        process="trainO"
        shift 2
        ;;
    "--testO")
        dataset_location="$curr_location/"$2
        echo "Reading the  data location :"$dataset_location
        process="testO"
        shift 2
        ;;
    # define general or other flags over here
    --b)
        flag_passed='-b'
        fuel_SMILE=$2
        echo  "Type of Bonds in $fuel_SMILE : " 
        shift 2
        ;;   
    --c) 
        error_criteria=$2
        echo "Defined criterion for error based clustering :  $error_criteria "
        echo
        shift 2
        ;; 
    --i) 
        max_iter=$2
        echo "Maximum iterations :  $error_criteria "
        echo
        shift 2
        ;;  
    --n) 
        num_clusters=$2
        echo "Number of clusters :  $num_clusters "
        echo
        shift 2
        ;; 
    --a)
        flag_passed='-a'
        echo "Manual Analysis of fuel-dataset given range of parameter:"
        analysis_type=$2
        shift 2
        ;;
    --"sfile")
        echo  "Analysis of file given:"
        dataset_location="$curr_location/$2"
        echo $dataset_location
        shift 2
        ;;
    --h)
        flag_passed='-h'
        echo "Histogram plots of parameters:"
        dataset_location="$curr_location/"$2
        echo
        shift 2
        ;;
    --k)
        flag_passed='-k'
        echo "External Data passsed to predict the Ignition Delay and to store result seperately"
        dataset_location="$curr_location/"$2
        echo
        shift 2
        ;;
    --f)
        flag_passed='-f'
        echo "Plotting of all test result"
        dataset_location="$curr_location/"$2
        echo
        shift 2
        ;;
    --p)  
        flag_passed='-p'
        echo "Plot of average coeffcient value obtained by histogram of coeffcients"
        dataset_location="$curr_location/"$2
        echo $dataset_location
        echo
        shift 2
        ;;
    --r)
        flag_passed='-r'
        echo "Eliminiation of feature set as True"
        elimination=$2
        echo
        shift 2
        ;;
    --l)
        flag_passed='-l'
        echo "Refrence Points are limited by features"
        limited_ref_points=$2
        echo $limited_ref_points
        shift 2
        ;;
    --s)
        flag_passed='-s'
        significance_level=$2
        echo "Significance level for backward elimination : $significance_level"
        echo
        shift 2
        ;;
    --d)
        flag_passed='-d'
        echo 'Generating dataset with transformed feature'
        dataset_location="$curr_location/"$2
        echo
        shift 2
        ;;
    *)
        echo ""
        echo "Wrong argument passed : "
        echo " --algo : Select algorithm Multiple regression, Spath, GMM, Error based tree"
        echo " --train : Select algorithm Multiple regression, Spath, GMM, Error based tree"
        echo " --test : Select algorithm Multiple regression, Spath, GMM, Error based tree"
        echo " --trainO : Select algorithm Multiple regression, Spath, GMM, Error based tree"
        echo " --testO : Select algorithm Multiple regression, Spath, GMM, Error based tree"
        echo " --sfile : Select algorithm Multiple regression, Spath, GMM, Error based tree"
        echo " --c : Critria for error based clustering"
        echo " --l : Limit of number of refrence point"
        echo " --b : Bond analysis in the Fuel "
        echo " --a : Analysis for fuel parameters"
        echo " --i : Maximum number of iteration"
        echo " --n : Maximum number of cluster"
        echo " --h : Histogram Plots"
        echo " --f : Plot histogram of all the result"
        echo " --k : External fuel analysis saving all the result"
        echo " --p : plotting of coefficient to find Average values"
        echo " --r : Backward elimination Activation True/False"
        echo " --s : Significance Level for Backward Elimination"
        echo " --d : Dataset generation ONLY for fuel contaning SMILES"
        echo ""
        break
        ;;
    esac
    num=$((num-1)) #reducing flag value
done



########################################################
################Common Function or Ploting #############
########################################################
plotting_fucntion () {
    #moving to plots folder

    cd "$curr_location/plots/"


    #gnerating pdf from tex file 
    # pdflatex Training.tex > /dev/null 2>&1  #to not print output 
    #opeing the pdf file 
    # xdg-open Training.pdf
    echo 'Plotting of Training done'

    # pdflatex Testing.tex > /dev/null 2>&1
    # xdg-open Testing.pdf
    # echo 'Plotting of Testing done'

    # pdflatex Datasize.tex > /dev/null 2>&1
    # xdg-open Datasize.pdf
    echo 'Plotting of Datasize done'

    # pdflatex MaxRelError.tex > /dev/null 2>&1
    # xdg-open MaxRelError.pdf
    echo 'Plotting of MaxRelError done'

    # pdflatex ChildLabel.tex > /dev/null 2>&1
    # xdg-open ChildLabel.pdf
    echo 'Plotting of Labels done'

    cd 
    cd $code_directory
    cd ./src/tree

    $runVar coef_tikz_compatible.py "$curr_location/plots/"
    # $runVar Fuel_tikz_compatible.py "$curr_location/plots/"

    dir_to_plot="$curr_location/plots/"
    cd $dir_to_plot
    # pdflatex coefficient.tex > /dev/null 2>&1
    # xdg-open coefficient.pdf
    echo 'Plotting of Coefficients done'

    #in plotting dir, deleting all files except .pdf
    find $dir_to_plot  -name '*.aux' -delete
    find $dir_to_plot  -name '*.tex' -delete
    find $dir_to_plot  -name '*.log' -delete


    # pdflatex FuelsTrainingTesting.tex > /dev/null 2>&1
    # xdg-open FuelsTrainingTesting.pdf
}



# ###############################################
# #BASED ON FLAG, IT WILL RUN THE $runVar SCRIPT#
# ###############################################

if [ $method == 'common' ]
then 
    echo "Working on common methods"
    ##Analysis of fuel dataset
    if [ $flag_passed == '-a' ]
    then
        if [ $analysis_type == 'manual' ]
        then 
            $runVar DDS.py -a $method $process $dataset_location  $curr_location $analysis_type
        fi

        if [ $analysis_type == 'temperature' ]
        then 
            $runVar DDS.py -a $method $process $dataset_location  $curr_location $analysis_type
        fi

        if [ $analysis_type == 'pressure' ]
        then 
            $runVar DDS.py -a $method $process $dataset_location  $curr_location $analysis_type
        fi

        if [ $analysis_type == 'parameter' ]
        then 
            $runVar DDS.py -a $method $process $dataset_location  $curr_location $analysis_type
        fi

        if [ $analysis_type == 'distribution' ]
        then 
            $runVar DDS.py -a $method $process $dataset_location  $curr_location $analysis_type
        fi
    fi

    if [ $flag_passed == '-d' ]
    then
    $runVar DDS.py -d $method $process $dataset_location  $curr_location
    fi

    if [ $flag_passed == '-b' ]
    then
    $runVar DDS.py -b $method $process $fuel_SMILE $curr_location #SMILE will passed in place of dataset location
    fi

    ##Histogram plots of data 
    if [ $flag_passed == '-h' ]
    then
    $runVar DDS.py -h $method $process $dataset_location $curr_location
    fi


    #################################
    #################################
    #other than fuel data set if dataset 
    #ready you just want to generate model
    #feature selection will be different

    if [ $flag_passed == '-k' ]
    then
    $runVar DDS.py -k $method $process $dataset_location $curr_location
    echo 'done'
    fi

    if [ $flag_passed == '-f' ]
    then
    $runVar DDS.py -f $method $process $dataset_location $curr_location
    echo 'done'
    fi


    if [ $flag_passed == '-p' ]
    then
    $runVar DDS.py -p $method $process $dataset_location $curr_location
    echo 'done'
    fi
fi

#################################################################################
#flag in methods are of no use it is just there to keep consistency in the format
#################################################################################

if [ $method == 'GMM' ]
then 
if [ $process == 'train' ]
    then
        echo 'GMM'
        #removing old directory
        rm -rf "$curr_location/object_file"
        rm -rf "$curr_location/plots"
        echo "Generating cluster using GMM"
        $runVar DDS.py -t $method $process $dataset_location $curr_location $num_clusters $max_iter $limited_ref_points
        # plotting_fucntion
    fi

    if [ $process == 'test' ]
    then
        #removing old directory
        echo "Testing the model using GMM"
        $runVar DDS.py -e $method $process $dataset_location $curr_location
        # plotting_fucntion
    fi
fi

if [ $method == 'tree' ]
then 
    if [ $process == 'train' ] 
    then
        echo 'tree'
        #removing old directory
        rm -rf "$curr_location/object_file"
        rm -rf "$curr_location/plots"
        echo "Generate cluster and Training/Testing the model based on tree"
        $runVar DDS.py -t $method $process $dataset_location $curr_location $error_criteria $elimination $significance_level $limited_ref_points
        # plotting_fucntion
    fi

    if [ $process == 'test' ] 
    then
        #removing old directory
        echo "Generate cluster and Training/Testing the model based on tree"
        $runVar DDS.py -e $method $process $dataset_location $curr_location
        # plotting_fucntion
    fi
fi

if [ $method == 'spath' ]
then 
    if [ $process == 'train' ]
    then
        echo 'tree'
        #removing old directory
        rm -rf "$curr_location/object_file"
        rm -rf "$curr_location/plots"
        echo "Generate cluster and Training/Testing the model based on Spath"
        $runVar DDS.py -t $method $process $dataset_location $curr_location $num_clusters $max_iter $limited_ref_points
        # plotting_fucntion
    fi

    if [ $process == 'test' ]
    then
        #removing old directory
        echo "Generate cluster and Training/Testing the model based on Spath"
        $runVar DDS.py -e $method $process $dataset_location $curr_location
        # plotting_fucntion
    fi
fi

if [ $method == 'multi' ] 
then 
    if [ $process == 'train' ]
    then
        echo 'tree'
        #removing old directory
        rm -rf "$curr_location/object_file"
        rm -rf "$curr_location/plots"
        echo "Running Multiple Linear Regression:"
        $runVar DDS.py -m $method $process $dataset_location $curr_location $elimination $significance_level
        # plotting_fucntion
    fi

    if [ $process == 'test' ] 
    then
        #removing old directory
        echo "Generate cluster and Training/Testing the model based on Spath"
        $runVar DDS.py -e $method $process $dataset_location $curr_location
        # plotting_fucntion
    fi
fi

echo 
echo
echo
echo "Processs Completed!! "
cd 
cd $code_directory
#removing file
rm filelocation.txt  
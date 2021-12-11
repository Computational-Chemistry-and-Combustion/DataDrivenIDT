chmod +x Run.sh
cd TryYourself/nAlkaneIDT/

# - echo 'Analyzing Data - Need User Input'
# - pwd>>$TRAVIS_BUILD_DIR/filelocation.txt;$TRAVIS_BUILD_DIR/Run.sh -a trainset.csv

echo 'Checking Bond Details for Heptane'
pwd>>$TRAVIS_BUILD_DIR/filelocation.txt
$TRAVIS_BUILD_DIR/Run.sh -b CCCCCCC

echo 'Generating Histograms of parameters for each fuel'
pwd>>$TRAVIS_BUILD_DIR/filelocation.txt
$TRAVIS_BUILD_DIR/Run.sh -h trainset.csv

echo 'Multiple regression'
pwd>>$TRAVIS_BUILD_DIR/filelocation.txt
$TRAVIS_BUILD_DIR/Run.sh -c 0.05 -r False -m trainset.csv

echo 'Multiple regression with significance and feature elimination'
pwd>>$TRAVIS_BUILD_DIR/filelocation.txt
$TRAVIS_BUILD_DIR/Run.sh -c 0.05 -r True -s 0.05 -m trainset.csv

pwd>>$TRAVIS_BUILD_DIR/filelocation.txt
$TRAVIS_BUILD_DIR/Run.sh -c 0.05 -t trainset.csv

echo 'External test'
pwd>>$TRAVIS_BUILD_DIR/filelocation.txt
$TRAVIS_BUILD_DIR/Run.sh -e testset.csv

cd ..
cd BostonHousingPrice/

echo 'Boston Hosing Price Data - Training and Testing'
pwd>>$TRAVIS_BUILD_DIR/filelocation.txt
$TRAVIS_BUILD_DIR/Run.sh -c 0.1 -o trainset.csv
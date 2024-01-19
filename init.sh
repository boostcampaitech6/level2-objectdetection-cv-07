# data 다운로드
if [ -d "../dataset" ] ; then
	echo -e "\e[34m'train' and 'eval' folders exist in the parent directory\e[0m"
else
	cd ..
	wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000266/data/data.tar.gz
	tar -zxvf data.tar.gz
	rm data.tar.gz
	cd level2-objectdetection-cv-07
fi
echo -e "\e[34mFin data download\e[0m"

# git 설정
git config --global commit.template ./.commit_template
git config --global core.editor "code --wait"
echo -e "\e[34mFin git config\e[0m"

# pre-commit 설정
pre-commit autoupdate
pre-commit install
echo -e "\e[34mFin pre-commit\e[0m"

# install requirements
pip install -r requirements.txt
echo -e "\e[34mFin install requirements\e[0m"

echo -e "\e[34mFin init\e[0m"
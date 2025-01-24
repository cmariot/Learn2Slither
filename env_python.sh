#!/bin/sh

# You must source this script to activate the virtualenv in your shell.
# source my_script.sh


ENV_NAME=.venv
BLUE='\033[0;34m'
NC='\033[0m'


PYTHON=python3.11

# Disable the Pygame output
export PYGAME_HIDE_SUPPORT_PROMPT="hide"

# Create a virtualenv based on the ENV_NAME variable.
$PYTHON -m venv $ENV_NAME
echo "${BLUE}${ENV_NAME} virtual environment has been created.${NC}"
echo

# Activate the virtualenv.
source $ENV_NAME/bin/activate
echo "${BLUE}The virtual environment is activated.${NC}"
echo

# Install the requirements in the virtualenv.
echo "${BLUE}Installing the requirements...${NC}"
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -r requirement.txt


echo

# Show the installed packages.
echo "${BLUE}Installed packages:${NC}"
$PYTHON -m pip freeze
echo

echo "${BLUE}Running the program...${NC}"

$PYTHON ./srcs/main.py

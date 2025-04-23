# Makefile for Vietnamese Fake News Detector Project

# You can override these if you want to use a custom Python or venv path
PYTHON ?= python3
PIP     ?= pip3

# Virtual environment directory (optional)
VENV_DIR := venv
ifeq ($(OS),Windows_NT)
  PLATFORM := windows
else
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
  else ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
  else
    PLATFORM := unknown
  endif
endif

ifeq ($(PLATFORM),windows)
  ACTIVATE := $(VENV_DIR)/Scripts/activate
else
  ACTIVATE := source $(VENV_DIR)/bin/activate
endif

# Requirements file
REQS := requirements.txt

.PHONY: all venv install lint format test crawl generate classify ui build-exe clean

all: install

## Create & activate a virtualenv, then install dependencies
venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(ACTIVATE) && $(PIP) install --upgrade pip

install: venv
	$(ACTIVATE) && $(PIP) install -r $(REQS)

## Run the crawler once
crawl:
	$(ACTIVATE) && $(PYTHON) vnexpress_crawler.py

## Generate fake news
generate:
	$(ACTIVATE) && $(PYTHON) fake_new_generator.py

## Train/evaluate classifier
classify:
	$(ACTIVATE) && $(PYTHON) fake_new_classifier.py

## Launch the Streamlit UI
ui:
	$(ACTIVATE) && streamlit run ui.py

## Auto-run daily crawler
daily-crawl:
	$(ACTIVATE) && $(PYTHON) daily_vnexpress_crawler.py

## Train LSMT Model
train-lstm:
	$(ACTIVATE) && $(PYTHON) new_training.py
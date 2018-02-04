PYTHON=python3.6


# ========== Linux ==========


# ----- Install -----

install:
	$(if $(shell apt-cache search $(PYTHON)), , \
		sudo add-apt-repository -y ppa:fkrull/deadsnakes && apt-get update)
	sudo apt-get install -y build-essential
	sudo apt-get install -y $(PYTHON) $(PYTHON)-dev python3-pip
	sudo apt-get install -y libpq-dev postgresql postgresql-contrib

# ----- Virtualenv -----

venv_init:
	if [ ! -d "venv" ]; then $(PYTHON) -m venv venv ; fi;
	bash -c "source venv/bin/activate && \
		pip install --upgrade wheel pip setuptools && \
		pip install --upgrade --requirement requirements.txt"

# ----- Setup -----

setup: install venv_init

# ----- Clean -----

clean:
	find . -path ./venv -prune -o -name "__pycache__" -exec rm -rf {} \;
	find . -path ./venv -prune -o -name "*.pyc" -exec rm -rf {} \;
	rm -rf .cache

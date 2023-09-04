PACKAGE_NAME = xalgorithm

.PHONY: all build update upload clean

all: build

build:
	python setup.py $(version) sdist

update: build 
	@echo "__version__ = '$(version)'"
	bash ./scripts/update_version.sh $(version)

upload: update
	twine upload dist/$(PACKAGE_NAME)-$(version).tar.gz --config-file .pypirc

clean:
	rm -rf dist

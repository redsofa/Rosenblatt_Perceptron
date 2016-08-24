GOPATH := $(shell pwd)


all: build

go-deps:
	@echo Using Go Path :
	@echo $(GOPATH)	
	@echo NO DEPENDENCIES TO FETCH

dist-dir:
	@echo Creating dist directory
	mkdir ./dist

go-doc:
	@echo Launching Go docs server
	godoc -http=:6060
	$ (cd src/github.com/redsofa/; godoc -http=:6060)

build: clean dist-dir 
	@echo Building perceptron
	@echo Using Go Path :
	@echo $(GOPATH)
	mkdir ./dist/perceptron
	go build -o ./dist/perceptron/perceptron src/github.com/redsofa/perceptron/main/main.go

dist: build
	@echo Distributing perceptron
	cp src/github.com/redsofa/perceptron/main/config.json ./dist/perceptron/config.json

clean:
	@echo Cleaning ...
	rm -fR ./dist
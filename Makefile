#!/bin/bash
# Makefile for Perceptron
CC = g++
DEBUG = -g
PROJNAME = perceptron

HEADERPATH = ./src
SOURCEPATH = ./src

LOCALDEPSINCLUDES = ./deps
AUXINCLUDES = 
AUXLIBS = 

INCLUDES = -I$(LOCALDEPSINCLUDES) -I$(AUXINCLUDES)  
LIBS = -L$(AUXLIBS) 
#LIBS += -L/usr/local/lib/
CFLAGS = -std=gnu++11 -std=c++11 -O3 -Wall  -fmessage-length=0 -fPIC $(INCLUDES)
CFLAGS += $(DEBUG)
LFLAGS = $(LIBS)
#For verbosity
LFLAGS += -v
LDFLAGS = -shared

HDRS  = $(shell find $(HEADERPATH) $(AUXINCLUDES) $(LOCALDEPSINCLUDES) -name '*.h')
HDRS += $(shell find $(HEADERPATH) $(AUXINCLUDES) $(LOCALDEPSINCLUDES) -name '*.h++')
SRCS  = $(shell find $(SOURCEPATH) -name '*.cpp')
SRCS += $(shell find $(SOURCEPATH) -name '*.c')
OBJS = $(SRCS:.cpp=.o)
TXTS = $(wildcard *.txt)
SCRIPTS = $(wildcard *.sh)

all : $(PROJNAME)
# all :$(PROJNAME).a $(PROJNAME).so

$(PROJNAME).a : $(OBJS)
	@echo Creating static lib $@
	ar rcs $@ $(OBJS)

$(PROJNAME).so : $(OBJS)
	@echo Creating dynamic lib $@
	$(CC) -o $@ $(OBJS) $(LDFLAGS) $(LFLAGS) 

%.o: %.cpp $(HDRS)
	$(CC) -c $(CFLAGS) $(LFLAGS) -o $@ $<

$(PROJNAME): $(OBJS)
	@echo Compiling program $@
	$(CC)  $^ $(CFLAGS) $(LFLAGS) -o $@

clean:
	@echo Clean
	rm -f *~ *.o *~
	@echo Success

cleanall:
	@echo Clean All
	rm -f *~ $(SOURCEPATH)/*.o *~ $(PROJNAME).a $(PROJNAME).so $(PROJNAME)
	@echo Success
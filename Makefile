#
# LISE package top-level Makefile.
#
# Edit file './LISE.defs' to the specifics of the target platform.
#
# Invocation(s) from './':
# 	make clean
# 	make -e 
#
# Successful builds will copy program executables to './bin'.
# A copy of the programs is left in the respective directories './LISE-SLDAsolver', './LISE-TDSLDA'. 
# 'clean' removes the objects and programs from these directories only, not from './bin'. 
# The programs in './bin' are overwritten on subsequent builds. 
#

SHELL = /bin/sh

include $(CURDIR)/LISE.defs

LISE_BIN_DIR=$(CURDIR)/bin
$(shell mkdir -p $(LISE_BIN_DIR))

SUBDIRS = LISE-SLDAsolver LISE-TDSLDA

subdirs:
	for dir in $(SUBDIRS); do \
	  $(MAKE) -C $$dir; \
	done

.PHONY : clean subdirs $(SUBDIRS)

clean: $(SUBDIRS)
	rm -f *.o x* 

$(SUBDIRS):
	for dir in $(SUBDIRS); do \
	  $(MAKE) -C $$dir clean; \
	done


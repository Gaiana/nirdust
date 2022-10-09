Installation
============

This is the recommended way to install NIRDust.

Installing with pip
--------------------

Make sure that you are using Python 3.8 or newer. The most convenient way 
to install NIRDust is within a virtual environment via the pip command.

After setting up and activating the virtualenv, run the following command:

.. code-block:: console

   $ pip install nirdust

Now NIRDust should be installed in your system along with all its dependencies.


Installing the development version
----------------------------------

If you’d like to be able to update your NIRDust copy with the latest bug
fixes and improvements, follow these instructions:

Make sure that you have Git installed and that you can run its commands from a shell.
(Enter ``git help`` at a shell prompt to test this.)

Check out nirdust main development branch as follows:

.. code-block:: console

   $ git clone https://github.com/Gaiana/nirdust

This will create a directory *nirdust* in your current directory.

Then you can proceed to install with the commands

.. code-block:: console

   $ cd nirdust
   $ pip install -e .



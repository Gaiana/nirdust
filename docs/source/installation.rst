Installation
============


This is the recommended way to install Nirdust.

Installing  with pip
^^^^^^^^^^^^^^^^^^^^

Make sure that the Python interpreter can load Nirdust code.
The most convenient way to do this is to use virtualenv, virtualenvwrapper, and pip.

After setting up and activating the virtualenv, run the following command:

.. code-block:: console

   $ pip install nirdust
   ...




Installing the development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you’d like to be able to update your nirdust code occasionally with the
latest bug fixes and improvements, follow these instructions:

Make sure that you have Git installed and that you can run its commands from a shell.
(Enter *git help* at a shell prompt to test this.)

Check out nirdust main development branch like so:

.. code-block:: console

   $ git clone https://github.com/Gaiana/nirdust
   ...

This will create a directory *nirdust* in your current directory.

Then you can proceed to install with the commands

.. code-block:: console

   $ cd nirdust
   $ pip install -e .
   ...



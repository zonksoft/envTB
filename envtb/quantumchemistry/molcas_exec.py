import os.path
import os
import shutil
import tempfile
import subprocess

from envtb.quantumchemistry.molcas_log_output import MolcasOutput
from envtb.quantumchemistry.molden import MoldenFile
from envtb.quantumchemistry.inporb import InpOrbReader

class MolcasInput:
    def __init__(self, name):
        self.commands = []
        self.guess_orbital_file = None
        self.name = name
        
    def input_file_name(self):
        return self.name + ".input"
        
    def add_command(self, command):
        self.commands.append(command)
        
    def add_guess_orbital_file(self, filename):
        self.guess_orbital_file = filename
    
    def write_input_files(self, dir_path):
        input_file_string = \
            '\n'.join([" &%s\n%s" % (cmd.command, cmd.build_string()) 
                       for cmd in self.commands])
        
        input_file = open(os.path.join(dir_path, self.name + ".input"), "w")
        
        if self.guess_orbital_file is not None:
            input_file.write("> LINK FORCE %s INPORB\n" % os.path.basename(self.guess_orbital_file))
        input_file.write(input_file_string)
        input_file.write('\n')
        input_file.close()
        
        if self.guess_orbital_file is not None:
            shutil.copyfile(self.guess_orbital_file, os.path.join(dir_path, os.path.basename(self.guess_orbital_file)))
                 

class MolcasInputSection:
    def __init__(self, command):
        self.command = command
        
    def build_string(self):
        return ''
        
class MolcasInputScf(MolcasInputSection):
    def __init__(self, uhf=False, iterations=None):
        self.command = 'scf'
        self.arguments = []
        
        if uhf is True:
            self.arguments.append('uhf')
        if iterations is not None:
            self.arguments.append('iter=%d' % iterations)
            
    def build_string(self):
        return '\n'.join(self.arguments)
    
class MolcasInputGateway(MolcasInputSection):
    def __init__(self, basis_sets):
        self.command = 'gateway'
        self.basis_sets = basis_sets
        
    def build_string(self):
        return '\n'.join([basis_set.build_string() for basis_set in self.basis_sets])
    
    
class MolcasBasisSet:
    def __init__(self, basis_set_id, coord_string):
        self.basis_set_id = basis_set_id
        self.coord_string = coord_string
        
    def build_string(self):
        return """basis set
%s
%s
end of basis""" % (self.basis_set_id, self.coord_string)

class MolcasExec:
    def __init__(self, molcas_executable_path, molcas_input):
        """
        molcas_input: a MolcasInput object.
        """
        self.molcas_input = molcas_input
        self.molcas_executable_path = molcas_executable_path
        self.working_directory = tempfile.mkdtemp(prefix="envtb_molcas")
        
        molcas_log_raw = self.__run()
        self.molcas_log = MolcasOutput(molcas_log_raw)
        
        self.molden_output = self.__get_molden_orbital_output()

    def __run(self):
        self.molcas_input.write_input_files(self.working_directory)
        molcas_log = self.__run_molcas(self.working_directory)
        return molcas_log
        
    def __get_molden_orbital_output(self):
        return MoldenFile(
            os.path.join(self.working_directory, 
                         self.molcas_input.name + ".scf.molden"))
        
    def __run_molcas(self, working_directory):
        args = [self.molcas_executable_path, self.molcas_input.input_file_name()]
        env = os.environ.copy()
        env['WorkDir'] = working_directory
        molcas_exec = subprocess.Popen(args, cwd=working_directory,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        molcas_output = molcas_exec.communicate()[0].split('\n')
        if molcas_exec.returncode != 0:
            log_file = open(os.path.join(working_directory, 'LOG'), "w")
            log_file.write('\n'.join(molcas_output))
            log_file.close()
            raise ValueError('Return code of Molcas was %i. Check the files in %s. Molcas Output was written to %s.' %
                             (molcas_exec.returncode, working_directory, os.path.join(working_directory, 'LOG')))
        return molcas_output
    

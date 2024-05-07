from typing import List, Literal, Optional, OrderedDict, Tuple
import pytorch as torch
import fastprop 

class fastpropSolubility(fastprop):
    def __init__(
        self,
        num_solute_representation_layers: int = 0,
        num_solvent_representation_layers: int = 0,
        concatenation_operation: Literal["concatenation", "multiplication", "subtraction"] = "concatenation",
        num_features: int = 1613,
        num_concatenation_layers: int = 0,
        learning_rate: float = 0.001,
    ):
        super().__init__(
            input_size=num_features,
            hidden_size=1,  # we will overwrite this
            fnn_layers=0,  # and this
            readout_size=1,  # and this
            num_tasks=1,  # actually equal to len(descriptors), but we don't want to see performance on each
            learning_rate=learning_rate,
            problem_type="regression",
            target_names=[]
        )
        del self.fnn
        del self.readout
        self.num_solute_representation_layers = num_solute_representation_layers
        self.num_solvent_representation_layers = num_solute_representation_layers
        self.concatenation_operation = concatenation_operation
        self.num_hidden_concatenation_layers = num_concatenation_layers  

        # solute 
        self.solute_module = []
        self.solute_module.append(torch.nn.Identity())
        for i in range(num_solute_representation_layers): #hidden layers
          self.solute_module.append(torch.nn.Linear(num_features, num_features))
          self.solute_module.append(torch.nn.ReLU())

        #solvent
        self.solvent_module = []
        self.solvent_module.append(torch.nn.Identity())
        for i in range(num_solvent_representation_layers): #hidden layers
          self.solvent_module.append(torch.nn.Linear(num_features, num_features))
          self.solvent_module.append(torch.nn.ReLU())

        #assemble modules
        self.solute_representation_module = torch.nn.Sequential(*self.solute_module)
        self.solvent_representation_module = torch.nn.Sequential(*self.solvent_module)

        #concatenated module
        self.concatenated_module = []
        if self.concatenation_operation == "concatenation": #size doubles if concatenated
          self.concat_features = 2*num_features + 1 #plus temperature
          self.concatenated_module.append(torch.nn.Linear(self.concat_features, self.concat_features))
          self.concatenated_module.append(torch.nn.ReLU())
        else:
          self.concat_features = num_features + 1 #plus temperature
          self.concatenated_module.append(torch.nn.Linear(self.concat_features, self.concat_features))
          self.concatenated_module.append(torch.nn.ReLU()) #input layer
        for i in range(num_concatenation_layers): #hidden layers
          self.concatenated_module.append(torch.nn.Linear(self.concat_features, self.concat_features))
          self.concatenated_module.append(torch.nn.ReLU())
        self.concatenated_representation_module = torch.nn.Sequential(*self.concatenated_module)

        #readout
        self.readout = torch.nn.Linear(self.concat_features, 1)
        self.save_hyperparameters()

    def forward(self, x):
        solute = x.solute
        solvent = x.solvent
        temp = x.temperature
        solute_representation = self.solute_representation_module(solute)
        solvent_representation = self.solvent_representation_module(solvent)

        if self.concatenation_operation == "concatenation":
            concatenated_representation = torch.cat((solute_representation, solvent_representation, temp), dim=1)
        elif self.concatenation_operation == "multiplication":
            concatenated_representation = solute_representation * solvent_representation
            concatenated_representation = torch.cat((concatenated_representation, temp), dim=1)
        elif self.concatenation_operation == "subtraction":
            concatenated_representation = solute_representation - solvent_representation
            concatenated_representation = torch.cat((concatenated_representation, temp), dim=1)
        output = self.concated_representation_module(concatenated_representation)
        y = self.readout(output)
        return y
from hls4ml.backends.fpga.fpga_types import (
    ArrayVariableConverter,
    InplaceStreamVariableConverter,
    StreamVariableConverter,
    VariableDefinition,
)

# region ArrayVariable


class BambuArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return '{type} {name}{suffix}[{shape}]'.format(
            type=self.type.name, name=self.name, suffix=name_suffix, shape=self.size_cpp()
        )


class BambuInplaceArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class BambuArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Bambu', definition_cls=BambuArrayVariableDefinition)


class BambuInplaceArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Bambu', definition_cls=BambuInplaceArrayVariableDefinition)


# endregion

# region StreamVariable


class BambuStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        if as_reference:  # Function parameter
            return f'hls::stream<{self.type.name}> &{self.name}{name_suffix}'
        else:  # Declaration
            return 'hls::stream<{type}> {name}{suffix}("{name}")'.format(
                type=self.type.name, name=self.name, suffix=name_suffix
            )


class BambuInplaceStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class BambuStreamVariableConverter(StreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Bambu', definition_cls=BambuStreamVariableDefinition)


# endregion

# region InplaceStreamVariable


class BambuInplaceStreamVariableConverter(InplaceStreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(
            type_converter=type_converter, prefix='Bambu', definition_cls=BambuInplaceStreamVariableDefinition
        )


# endregion


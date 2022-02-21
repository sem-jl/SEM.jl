function print_field_types(io::IO, struct_instance)
    fields = fieldnames(typeof(struct_instance))
    types = [typeof(getproperty(struct_instance, field)) for field in fields]
    field_types = string.(fields).*": ".*string.(types)
    field_types = "   ".*string.(field_types).*("\n")
    print(io, field_types...)
end

function print_field_names(io::IO, struct_instance)
    fields = fieldnames(typeof(struct_instance))
    types = [nameof(typeof(getproperty(struct_instance, field))) for field in fields]
    field_types = string.(fields).*": ".*string.(types)
    field_types = "   ".*string.(field_types).*("\n")
    print(io, field_types...)
end

function print_type_name(io::IO, struct_instance)
    print(io, nameof(typeof(struct_instance))); print(io, "\n")
end

function print_type(io::IO, struct_instance)
    print(io, typeof(struct_instance)); print(io, "\n")
end
# Intentionally does not import .countdown_code.
#
# verl loads that module by file path (as "custom_module") rather than as part of
# this package. Importing it here as well would execute its @register("countdown_code")
# decorator on a second, distinct class object, and verl's registry raises
# ValueError when a name is registered twice with differing classes.

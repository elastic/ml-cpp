# Machine Learning: Best Practices for C++ Development


## Table of Contents
1. [Introduction](#introduction)
2. [Naming Conventions](#naming-conventions)
3. [Project and Library Structure](#project-and-library-structure)
4. [File Structure](#file-structure)
5. [Class Structure](#class-structure)
6. [Language Fundamentals](#language-fundamentals)
7. [Language Extensions](#language-extensions)
8. [Documentation](#documentation)
9. [Testing](#testing)
10. [General Comments](#general-comments)

## Introduction

The intent of this document is to present rules and guidelines detailing best practices for developers writing C++ code for Machine Learning at Elastic. It is not intended to be an exhaustive, definitive set of coding standards covering style, format etc. nor is it intended to replace careful study of "Best Practice" handbooks such as the "Effective C++" series of books by Scott Meyers.

The rules contained in this document have deliberately been made as generic as possible.  They should be agnostic to development platform, compiler type and version etc. with the caveat that the compiler supports the C++11 standard as a minimum. If a rule applies to a C++ standard version higher than this it will be specified as e.g. **_From C++14_.**

The guiding principle should be that consistency with existing code is paramount, therefore in the interests of brevity, detailed justification for the rules has been omitted.

The use of the words SHOULD, MUST etc., comply with RFC 2119.


## Naming Conventions

1.  Code SHOULD be consistent with its surrounding context.
1.  Names MUST be meaningful and concise
1.  Variable names MUST reflect use not type
1.  Local variables MUST be named as per `variableName`
1.  Classes, structs and unions MUST be named as per `CClassName`, `SStructName`, `UUnionName` respectively
    1.  Implementation files MUST have a `.cc` extension
    1.  Header files MUST have a `.h` extension
1.  Member variables MUST be named as per `m_ClassMember`,` s_StructMember`, `u_UnionMember`
	 For class, structure and union member variables respectively
1.  Static members MUST be named as per `ms_ClassStatic`, `ss_StructStatic`
1.  Methods SHOULD be named as per `methodName`
1.  Template classes MUST be named as per `CClassName`, `<typename TYPE_NAME>`
1.  Enumerations MUST be named as per `EEnumName`, `E_MemberName`
1.  Type aliases MUST be named as per `TTypeName`
1.  Type aliases referring to a template SHOULD identify the template instantiation concisely (we have a consistent naming convention in such cases which SHOULD be followed) e.g.

 	```
	using TDoubleVec = std::vector<double>
	using TSizeDoubleMap = std::map<std::size_t, double>
 	```
1.  Constants MUST be named as per `CONSTANT_NAME`
1.  Macros MUST be named as per `MACRO_NAME(...)`. However...
1. Macros SHOULD NOT be used unless unavoidable
1.  Files MUST be named as per `CClassName.cc`, `CClassName.h` if they contain a single or principle class named `CClassName`
1.  Files containing primarily global typedefs  SHOULD be named as per `<Identifier>Types.h` where `<Identifier> `pertains to the file contents.
1.  Files containing primarily constants SHOULD be named` Constants.h`
1.  Files containing the function `main()` MUST be called `Main[Xxx].cc`. Where `Xxx` is only used if necessary to distinguish multiple such files in the same directory
1.  Non-boolean accessor functions MUST be named as `clientId` NOT `getClientId`
1.  Boolean accessor functions MUST be named as `isComplete` NOT `complete`
1.  Parameter names in function declaration and definition MUST be identical
1.  Parameters in constructor initialiser lists SHOULD be as per `classMember`

## Project and Library Structure


1.  Existing namespace usage MUST be followed:
    Production code in a subdirectory `foo` resides in namespace `foo` or `foo_t` (or a nested namespace)
1.  Namespaces MUST be used for logical groupings of code
1.  Namespaces MUST NOT span libraries
1.  Namespaces SHOULD NOT be imported with `using` directives (like `using namespace ml`;), except in unit test implementation files.
    In particular, `using namespace std;` and `using namespace boost;` MUST NOT be used anywhere
1.  Files SHOULD be kept short
1.  Multiple classes defined in a single file SHOULD be avoided, except where they pertain to closely related functionality
1.  Shared constants and typedefs MUST be in a 'Types' namespace of the form
    ```
    <library namespace>_t
    ```
1.  Libraries MUST NOT have circular dependencies
1.  Platform specific code SHOULD be in a separate file of the form
    ```
    <class name>_<platform name>.cc
    ```


## File Structure


1.  All source files MUST be formatted with the `clang-format` tool  prior to check in.
    1.  This procedure can be simplified by use of the [dev-tools/clang-format.sh](dev-tools/clang-format.sh) script.
    1.  The same specific version of `clang-format` used by the [clang-format.sh](dev-tools/clang-format.sh) script MUST be used. It is recommended that this be obtained from the pre-built binary packages of LLVM available from http://releases.llvm.org/download.html
1.  The standard header file layout MUST be observed

    Header files MUST contain the following items in the order defined below
    1.  [Copyright statement](copyright_code_header.txt)
    1.  Include guard of the form
        ```
        #ifdef INCLUDED_[<namespace>_]<class name>_h
        ```
        Note that test files are in the global namespace and hence the part in square bracket should be omitted in this case.
    1.  Include files SHOULD be avoided in header files.
    1.  Include files SHOULD be in the recommended order (see below) if present
    1.  Forward declarations. These MUST be used wherever possible to reduce include file requirements
    1.  Class declarations. These MUST be in the recommended order (see below)
    1.  End of include guard. This SHOULD be followed by a comment indicating the name of the guard to which this pertains.
    1.  Judicious use of blank lines SHOULD be used to separate each of the above items.
1.  The standard implementation file layout MUST be observed.
	Implementation files MUST contain the following items in the order defined below
    1.  Elastic commercial code file header
    1.  Class include file
    1.  Other include files, in the recommended order (see below)
    1.  Unnamed namespace local declarations (use of this is preferred to private declarations in the header file)
    1.  Beginning of namespace for this library/application
    1.  Constructor Implementation
    1.  Destructor Implementation
    1.  Copy/Move constructors (if present)
    1.  Class operators (if present)
    1.  Other method implementations
    1.  End of namespace for this library/application
    1.  Judicious use of blank lines SHOULD be used to separate each of the above items.
1.  Standard ordering of `#include` statements SHOULD be followed
    1.  Own include file - for `.cc` files including their own `.h`
    1.  Other ML include files
    1.  3<sup>rd</sup> party library include files (including Boost)
    1.  Standard C++ include files
    1.  Standard C include files. However C++ header wrappers SHOULD be included in preference to the equivalent C header, e.g include `cstdlib` in preference to `stdlib.h`
1.  Include files SHOULD be grouped by library/subdirectory, with a blank line between each grouping. `clang-format` is then able to sort in alphabetical order within each grouped section.  It is best practice to list the ML include files in the build order of the libraries they relate to, as this helps to catch accidental circular dependencies.

## Class Structure


1.  Class headers SHOULD be broken into sections

    Elements SHOULD be placed in sections according to a number of criteria:
    1.  Scope: public, protected, private or hidden (private + unimplemented functions)
    1.  Type: constants, typedefs, methods, variables, or nested classes
    1.  Static or non static
1.  Header sections MUST be ordered according to scope - public, protected, private
1.  Each section in a class MUST be prefixed with its scope keyword (public/protected/private) even if this repeats the scope already in effect
1.  Structs and Unions MUST NOT be used for encapsulation
    1.  Access specifiers MUST NOT appear
    1.  Constructors (if present) MUST be trivial
    1.  Other methods SHOULD NOT be used. This includes any explicit destructor.
1.  Declarations SHOULD be given minimal scope
    1.  This is a general rule, that should guide the scoping of classes, data, enums and functions
    1.  It has a number of corollaries:
        1.  Class variable data MUST NOT be public - this rule does not apply to static constants
        1.  A class which is used by only one other class SHOULD be nested inside it
        1.  A class which is instantiated in only one function MAY be nested inside the function definition.  A lambda SHOULD be used in preference in this case.
        1.  Typedefs, enums and static constants SHOULD be given minimal scope
1.  Copy constructor and assignment operator SHOULD be hidden unless appropriate - new classes SHOULD use `delete` for this purpose
1.  Functions SHOULD NOT use default arguments
1.  Functions and variable data not belonging to a class SHOULD be defined in a detail or unnamed namespace within implementation files
1.  Destructor of a polymorphic class MUST be virtual
1.  Destructor of a non-polymorphic class MUST NOT be virtual
1.  Interdependencies between static objects MUST be avoided
1.  Headers MUST NOT define static variables of non-built-in types
1.  Implementation details SHOULD reside in the `.cc` file
1.  Template code SHOULD either be in implementation file, or inside class declaration
1.  Function parameters MUST be ordered as [in] [in,out] [out]
1.  Classes SHOULD be forward declared rather than included
1.  Logically const methods SHOULD be const
1.  Headers SHOULD NOT define static variables of non-built-in types

## Language Fundamentals

1.  `nullptr` MUST be used in preference to `0` or `NULL`
1.  Unreachable or ineffectual code MUST NOT be included (this also applies to 'commented out' code)
1.  Functions returning a value MUST return a default value at the end
1.  Exceptions SHOULD NOT be thrown - use return codes to explicitly handle error conditions
1.  Exceptions thrown from 3rd party code MUST be caught in the smallest enclosing scope - such exceptions MUST be converted to an appropriate error code
1.  Multi-value error codes SHOULD be returned as an enumeration
1.  Switch statements SHOULD switch over enums and MUST cover all the cases
    1.  When switching over a non-enum a default case SHOULD be used
    1.  When switching over an enum a default case SHOULD NOT be used
1.  Error conditions SHOULD always be returned early
1.  The declaration of all variables SHOULD be given the minimal possible scope
1.  Return codes SHOULD preserve the abstraction and encapsulation of the class - functions should not return an error specific to the internal implementation
1.  `assert()` MUST NOT be used
1.  C-style casts MUST NOT be used
1.  Macros SHOULD NOT be used
1.  Objects and references SHOULD be used in preference to pointers
1.  Smart pointers SHOULD be used in preference to raw pointers
1.  Dynamically acquired resources SHOULD be released in the same scope
1.  A class wrapper SHOULD be used to avoid dangling resources
1.  Resource ownership MUST be indicated by parameter type

       Argument, Passed By                         | Memory Ownership
       ------------------------------------------- | ----------------
       Value                                       | N/A
       Pointer                                     | Called function
       Const pointer (not pointer to const object) | Calling function
       Reference                                   | Calling function
       Const reference                             | Calling function

1.  Explicit integer definitions, specifying size, SHOULD be used
1.  Member functions MUST be scoped with `this->` when called
1.  Floating point variables SHOULD be `double`
1.  Lambdas SHOULD be used in preference to any form of `bind`
1.  Default lambda capture modes SHOULD be avoided
1.  The `override` keyword SHOULD be used consistently within a source file
1.  Type aliases MUST be used in preference to typedefs - use `using` to create a type alias not `typedef`
1.  Rvalue references SHOULD only be used in the following cases
    1.  For implementing move semantics
    1.  For forwarding references in template code
1.  Emplace operations SHOULD be used to add items to containers wherever applicable i.e. prefer `emplace_back` over `push_back` for vectors and `emplace` over `insert` for maps
1.  Containers SHOULD have their capacity reserved in advance if applicable
1.  Range based for loops SHOULD be preferred over their explicit counterparts
1.  Uniform (braced) initialisers SHOULD be preferred in new code units
1.  The `auto` keyword SHOULD be used liberally for variable assignments when the resulting code is less verbose
1.  `auto` SHOULD NOT be used where the assigned type is not clear in the context e.g.
	```
	auto obj = doSomething(someVariable); // bad
	```
1. `auto` types should always be assigned with `operator=`

## Language Extensions

1.  Language features MUST NOT be used until supported by all compilers used _for a given ML build version_
    1.  The current lowest common denominator compiler version is Visual Studio 2013
    1.  C++14 features may be used as of version 7.0
1.  Where both Boost and the standard library contain equivalent implementations of the same feature the same provider SHOULD be used consistently across the codebase
1.  `make_shared` &  <code>make_unique (<strong><em>From C++14</em></strong>)</code> SHOULD be used to create the corresponding smart pointer types
1.  Use of 3rd party libraries (including Boost) MUST be approved - this applies to new features used for the first time.
1.  STL algorithms SHOULD be used wherever appropriate

## Documentation

1.  Source code MUST be readable. This is of primary importance
1.  Doxygen MUST be used to comment all source files
1.  Header files MUST be commented for Doxygen in standard format.
    The following MUST be included for all top-level classes:
    1.  Brief summary
    1.  Detailed description - what the class does.
    1.  Implementation decisions - what has been done and why.
    1.  In addition, the following MAY be used if required:
    1.  Future enhancements - what we should do to this class in the future.
    1.  Resource ownership - should be used where this class manages any resources (e.g. objects on the heap)
    1.  Example - usage example etc
1.  Exclamation mark style MUST be used for Doxygen
1.  Implementation files SHOULD be commented in C++ style (not C, and not Doxygen)
1.  All class members SHOULD be documented
1.  All public and protected methods MUST be documented with `\param, \return`
1.  Parameters MUST be documented by name
1.  Non-trivial resource ownership SHOULD be documented

## Testing

The CPPUNIT framework is used for testing the Elastic Machine Learning C++ code.

1.  Every class SHOULD have a corresponding unit test suite
1.  Test classes SHOULD belong to the parent or global namespace
1.  Unit tests SHOULD exist for every public method in the corresponding class

## General Comments

1.  The number of lines in a method SHOULD be kept to a minimum
1.  C++ functions SHOULD be used in preference to C
1.  `reinterpret_cast` MUST only be used when interfacing with 3<sup>rd</sup> party code
1.  `dynamic_cast` SHOULD be used judiciously
1.  `const_cast` SHOULD be used judiciously
1.  Classes SHOULD NOT inherit from more than one base class
1.  All code SHOULD compile with no warnings
1.  Overloaded function parameters SHOULD NOT be implicitly convertible

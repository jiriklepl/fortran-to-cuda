! The residency domain of an application that keeps no residency state of its own.
! Entries and directives of such an application use this one object.
module fort_resident_application
  use fort_resident_domain, only: resident_domain, resident_require
  implicit none
  private
  public :: resident, resident_require
  type(resident_domain), save, target :: resident
end module

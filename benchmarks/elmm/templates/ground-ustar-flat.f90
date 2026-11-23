  subroutine GroundUstarFlat(res, n)
    real(knd), intent(inout) :: res
    integer, intent(inout) :: n
    integer :: k
    !$fort reduction(+: res, n)
    do k = 1, size(UzmWMpoints)
      if (UzmWMpoints(k)%zk==1) then
        res = res + UzmWMpoints(k)%ustar
        n = n + 1
      end if
    end do
    !$fort reduction(+: res, n)
    do k = 1, size(UzpWMpoints)
      if (UzpWMpoints(k)%zk==1) then
        res = res + UzpWMpoints(k)%ustar
        n = n + 1
      end if
    end do
    !$fort reduction(+: res, n)
    do k = 1, size(VzmWMpoints)
      if (VzmWMpoints(k)%zk==1) then
        res = res + VzmWMpoints(k)%ustar
        n = n + 1
      end if
    end do
    !$fort reduction(+: res, n)
    do k = 1, size(VzpWMpoints)
      if (VzpWMpoints(k)%zk==1) then
        res = res + VzpWMpoints(k)%ustar
        n = n + 1
      end if
    end do
  end subroutine GroundUstarFlat



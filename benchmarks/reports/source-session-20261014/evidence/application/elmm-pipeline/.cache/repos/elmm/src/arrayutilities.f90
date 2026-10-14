! module MyBLAS
!   !Warning, all supplied arrays must be contiguous! This attribute not used due to compatibility with Oracle Solrais Studio 12.3
! 
!   use iso_c_binding
! 
!   interface
!     subroutine saxpy_(n,a,px,incx,py,incy) bind(C,name="saxpy_")
!       import
!       integer,intent(in)      :: n
!       real(c_float),intent(in) :: a
!       integer(c_intptr_t),value       :: px, py
!       integer,intent(in)      :: incx, incy
!     end subroutine
! 
!     subroutine daxpy_(n,a,px,incx,py,incy) bind(C,name="daxpy_")
!       import
!       integer,intent(in)        :: n
!       real(c_double),intent(in) :: a
!       integer(c_intptr_t),value         :: px, py
!       integer,intent(in)        :: incx, incy
!     end subroutine
!   end interface
! 
!   interface axpy
!     module procedure saxpy1D
!     module procedure daxpy1D
!     module procedure saxpy2D
!     module procedure daxpy2D
!     module procedure saxpy3D
!     module procedure daxpy3D
!   end interface
! 
!   contains
! 
!     subroutine saxpy1D(a, x, y)
!       real(c_float),target :: a, x(:), y(:)
!       integer(c_intptr_t) :: px, py
!       !y = y + a * x
!       px = loc(x(1))
!       py = loc(y(1))
!       call saxpy_(size(x),a,px,1,py,1)
!     end subroutine
! 
!     subroutine daxpy1D(a, x, y)
!       real(c_double),target :: a, x(:), y(:)
!       integer(c_intptr_t) :: px, py
!       !y = y + a * x
!       px = loc(x(1))
!       py = loc(y(1))
!       call daxpy_(size(x),a,px,1,py,1)
!     end subroutine
! 
!     subroutine saxpy2D(a, x, y)
!       real(c_float),target :: a, x(:,:), y(:,:)
!        integer(c_intptr_t) :: px, py
!      !y = y + a * x
!       px = loc(x(1,1))
!       py = loc(y(1,1))
!       call saxpy_(size(x),a,px,1,py,1)
!     end subroutine
! 
!     subroutine daxpy2D(a, x, y)
!       real(c_double),target :: a, x(:,:), y(:,:)
!       integer(c_intptr_t) :: px, py
!       !y = y + a * x
!       px = loc(x(1,1))
!       py = loc(y(1,1))
!       call daxpy_(size(x),a,px,1,py,1)
!     end subroutine
! 
!     subroutine saxpy3D(a, x, y)
!       real(c_float),target :: a, x(:,:,:), y(:,:,:)
!       integer(c_intptr_t) :: px, py
!       !y = y + a * x
!       px = loc(x(1,1,1))
!       py = loc(y(1,1,1))
!       call saxpy_(size(x),a,px,1,py,1)
!     end subroutine
! 
!     subroutine daxpy3D(a, x, y)
!       real(c_double),target :: a, x(:,:,:), y(:,:,:)
!       integer(c_intptr_t) :: px, py
!       !y = y + a * x
!       px = loc(x(1,1,1))
!       py = loc(y(1,1,1))
!       call daxpy_(size(x),a,px,1,py,1)
!     end subroutine
! 
! end module MyBLAS



module ArrayUtilities
  use Kinds
!   use MyBLAS

  implicit none

  private
  public exchange_alloc, assign, add, &
         set, multiply, add_multiplied, &
         multiply_and_add_scalar, reciprocal, &
         add_element, avg, dist, cross_product

  interface exchange_alloc
    module procedure exchange_alloc_1D
    module procedure exchange_alloc_2D
    module procedure exchange_alloc_3D
  end interface

  interface assign
    module procedure assign_1D
    module procedure assign_2D
    module procedure assign_3D
    module procedure assign_4D
  end interface

  interface add
    module procedure add_1D
    module procedure add_2D
    module procedure add_3D
    module procedure add_4D
    module procedure add_scalar_1D
    module procedure add_scalar_2D
    module procedure add_scalar_3D
  end interface

  interface set
    module procedure set_1D
    module procedure set_2D
    module procedure set_3D
    module procedure set_4D
    module procedure set_int_1D
    module procedure set_int_2D
    module procedure set_int_3D
    module procedure set_int_4D
  end interface

  interface multiply
    module procedure multiply_1D
    module procedure multiply_2D
    module procedure multiply_3D
  end interface

  interface add_multiplied
    module procedure add_multiplied_1D
    module procedure add_multiplied_2D
    module procedure add_multiplied_3D
    module procedure add_multiplied_4D
  end interface

  interface multiply_and_add_scalar
    module procedure multiply_and_add_scalar_1D
    module procedure multiply_and_add_scalar_2D
    module procedure multiply_and_add_scalar_3D
  end interface

  interface reciprocal
    module procedure reciprocal_1D
    module procedure reciprocal_2D
    module procedure reciprocal_3D
  end interface
  
  interface add_element
    module procedure add_element_int
  end interface
  
  interface avg
    module procedure avg_real_1D
    module procedure avg_real_2D
    module procedure avg_real_3D
    module procedure avg_int_1D
    module procedure avg_int_2D
    module procedure avg_int_3D
  end interface

  contains

    subroutine exchange_alloc_1D(A,B)
      real(knd),allocatable,intent(inout) :: A(:),B(:)
      real(knd),allocatable :: tmp(:)

      call move_alloc(A,tmp)
      call move_alloc(B,A)
      call move_alloc(tmp,B)
    end subroutine
    
    subroutine exchange_alloc_2D(A,B)
      real(knd),allocatable,intent(inout) :: A(:,:),B(:,:)
      real(knd),allocatable :: tmp(:,:)

      call move_alloc(A,tmp)
      call move_alloc(B,A)
      call move_alloc(tmp,B)
    end subroutine
    
    subroutine exchange_alloc_3D(A,B)
      real(knd),allocatable,intent(inout) :: A(:,:,:),B(:,:,:)
      real(knd),allocatable :: tmp(:,:,:)

      call move_alloc(A,tmp)
      call move_alloc(B,A)
      call move_alloc(tmp,B)
    end subroutine


    subroutine assign_1D(to,from)
      real(knd),contiguous,intent(out) :: to(:)
      real(knd),contiguous,intent(in)  :: from(:)
      integer :: i

      !$omp parallel do
      do i=1,size(to)
        to(i) = from(i)
      end do
      !$omp end parallel do
    end subroutine

    subroutine assign_2D(to,from)
      real(knd),contiguous,intent(out) :: to(:,:)
      real(knd),contiguous,intent(in)  :: from(:,:)
      integer :: j

      !$omp parallel do
      do j=1,size(to,2)
        to(:,j) = from(:,j)
      end do
      !$omp end parallel do
    end subroutine

    subroutine assign_3D(to,from)
      real(knd),contiguous,intent(out) :: to(:,:,:)
      real(knd),contiguous,intent(in)  :: from(:,:,:)
      integer :: k

      !$omp parallel do
      do k=1,size(to,3)
        to(:,:,k) = from(:,:,k)
      end do
      !$omp end parallel do
    end subroutine

    subroutine assign_4D(to,from)
      real(knd),contiguous,intent(out) :: to(:,:,:,:)
      real(knd),contiguous,intent(in)  :: from(:,:,:,:)
      integer :: k

      do k=1,size(to,4)
        call assign(to(:,:,:,k),from(:,:,:,k))
      end do
    end subroutine


    subroutine add_1D(to,from)
      real(knd),contiguous,intent(inout) :: to(:)
      real(knd),contiguous,intent(in)  :: from(:)
      integer :: i

      !$omp parallel do
      do i=1,size(to)
        to(i) = to(i) + from(i)
      end do
      !$omp end parallel do
    end subroutine

    subroutine add_2D(to,from)
      real(knd),contiguous,intent(inout) :: to(:,:)
      real(knd),contiguous,intent(in)  :: from(:,:)
      integer :: j

      !$omp parallel do
      do j=1,size(to,2)
        to(:,j) = to(:,j) + from(:,j)
      end do
      !$omp end parallel do
    end subroutine

    subroutine add_3D(to,from)
      real(knd),contiguous,intent(inout) :: to(:,:,:)
      real(knd),contiguous,intent(in)  :: from(:,:,:)
      integer :: k

      !$omp parallel do
      do k=1,size(to,3)
        to(:,:,k) = to(:,:,k) + from(:,:,k)
      end do
      !$omp end parallel do
!       call axpy(1._knd,from,to)
    end subroutine

    subroutine add_4D(to,from)
      real(knd),contiguous,intent(inout) :: to(:,:,:,:)
      real(knd),contiguous,intent(in)  :: from(:,:,:,:)
      integer :: k

      do k=1,size(to,4)
        call add(to(:,:,:,k),from(:,:,:,k))
      end do
    end subroutine


    subroutine add_scalar_1D(to,from)
      real(knd),contiguous,intent(inout) :: to(:)
      real(knd),intent(in)  :: from
      integer :: i

      !$omp parallel do
      do i=1,size(to)
        to(i) = to(i) + from
      end do
      !$omp end parallel do
    end subroutine

    subroutine add_scalar_2D(to,from)
      real(knd),contiguous,intent(inout) :: to(:,:)
      real(knd),intent(in)  :: from
      integer :: j

      !$omp parallel do
      do j=1,size(to,2)
        to(:,j) = to(:,j) + from
      end do
      !$omp end parallel do
    end subroutine

    subroutine add_scalar_3D(to,from)
      real(knd),contiguous,intent(inout) :: to(:,:,:)
      real(knd),intent(in)  :: from
      integer :: k

      !$omp parallel do
      do k=1,size(to,3)
        to(:,:,k) = to(:,:,k) + from
      end do
      !$omp end parallel do
    end subroutine


    subroutine set_1D(to,from)
      real(knd),contiguous,intent(out) :: to(:)
      real(knd),intent(in)  :: from
      integer :: i

      !$omp parallel do
      do i=1,size(to)
        to(i) = from
      end do
      !$omp end parallel do
    end subroutine

    subroutine set_2D(to,from)
      real(knd),contiguous,intent(out) :: to(:,:)
      real(knd),intent(in)  :: from
      integer :: j

      !$omp parallel do
      do j=1,size(to,2)
        to(:,j) = from
      end do
      !$omp end parallel do
    end subroutine

    subroutine set_3D(to,from)
      real(knd),contiguous,intent(out) :: to(:,:,:)
      real(knd),intent(in)  :: from
      integer :: k

      !$omp parallel do
      do k=1,size(to,3)
        to(:,:,k) = from
      end do
      !$omp end parallel do
    end subroutine
    
    subroutine set_4D(to,from)
      real(knd),contiguous,intent(out) :: to(:,:,:,:)
      real(knd),intent(in)  :: from
      integer :: k

      do k=1,size(to,4)
        call set(to(:,:,:,k), from)
      end do
    end subroutine
    

    subroutine set_int_1D(to,from)
      real(knd),contiguous,intent(out) :: to(:)
      integer,intent(in)  :: from
      integer :: i

      !$omp parallel do
      do i=1,size(to)
        to(i) = from
      end do
      !$omp end parallel do
    end subroutine

    subroutine set_int_2D(to,from)
      real(knd),contiguous,intent(out) :: to(:,:)
      integer,intent(in)  :: from
      integer :: j

      !$omp parallel do
      do j=1,size(to,2)
        to(:,j) = from
      end do
      !$omp end parallel do
    end subroutine

    subroutine set_int_3D(to,from)
      real(knd),contiguous,intent(out) :: to(:,:,:)
      integer,intent(in)  :: from
      integer :: k

      !$omp parallel do
      do k=1,size(to,3)
        to(:,:,k) = from
      end do
      !$omp end parallel do
    end subroutine
    
    subroutine set_int_4D(to,from)
      real(knd),contiguous,intent(out) :: to(:,:,:,:)
      integer,intent(in)  :: from
      integer :: k

      do k=1,size(to,4)
        call set(to(:,:,:,k),from)
      end do
    end subroutine
    

    subroutine multiply_1D(to,a)
      real(knd),contiguous,intent(inout) :: to(:)
      real(knd),intent(in)  :: a
      integer :: i

      !$omp parallel do
      do i=1,size(to)
        to(i) = to(i) * a
      end do
      !$omp end parallel do
    end subroutine

    subroutine multiply_2D(to,a)
      real(knd),contiguous,intent(inout) :: to(:,:)
      real(knd),intent(in)  :: a
      integer :: j

      !$omp parallel do
      do j=1,size(to,2)
        to(:,j) = to(:,j) * a
      end do
      !$omp end parallel do
    end subroutine

    subroutine multiply_3D(to,a)
      real(knd),contiguous,intent(inout) :: to(:,:,:)
      real(knd),intent(in)  :: a
      integer :: k

      !$omp parallel do
      do k=1,size(to,3)
        to(:,:,k) = to(:,:,k) * a
!         call sscal(size(to,2)*size(to,1),a,to,1)
      end do
      !$omp end parallel do
    end subroutine


    subroutine add_multiplied_1D(to,from,a)
      real(knd),contiguous,intent(inout) :: to(:)
      real(knd),contiguous,intent(in)  :: from(:)
      real(knd),intent(in)  :: a
      integer :: i

      !$omp parallel do
      do i=1,size(to)
        to(i) = to(i) + from(i) * a
      end do
      !$omp end parallel do
    end subroutine

    subroutine add_multiplied_2D(to,from,a)
      real(knd),contiguous,intent(inout) :: to(:,:)
      real(knd),contiguous,intent(in)  :: from(:,:)
      real(knd),intent(in)  :: a
      integer :: j

      !$omp parallel do
      do j=1,size(to,2)
        to(:,j) = to(:,j) + from(:,j) * a
      end do
      !$omp end parallel do
    end subroutine

    subroutine add_multiplied_3D(to,from,a)
      real(knd),contiguous,intent(inout) :: to(:,:,:)
      real(knd),contiguous,intent(in)  :: from(:,:,:)
      real(knd),intent(in)  :: a
      integer :: k

      !$omp parallel do
      do k=1,size(to,3)
        to(:,:,k) = to(:,:,k) +from(:,:,k) * a
      end do
      !$omp end parallel do
!        call axpy(a,from,to)
    end subroutine

    subroutine add_multiplied_4D(to,from,a)
      real(knd),contiguous,intent(inout) :: to(:,:,:,:)
      real(knd),contiguous,intent(in)  :: from(:,:,:,:)
      real(knd),intent(in)  :: a
      integer :: k

      do k=1,size(to,4)
        call add_multiplied(to(:,:,:,k),from(:,:,:,k),a)
      end do
    end subroutine


    subroutine multiply_and_add_scalar_1D(to,b,a)
      ! to = b * to + a
      real(knd),contiguous,intent(inout) :: to(:)
      real(knd),intent(in)  :: b
      real(knd),intent(in)  :: a
      integer :: i

      !$omp parallel do
      do i=1,size(to)
        to(i) = b * to(i) + a
      end do
      !$omp end parallel do
    end subroutine

    subroutine multiply_and_add_scalar_2D(to,b,a)
      ! to = b * to + a
      real(knd),contiguous,intent(inout) :: to(:,:)
      real(knd),intent(in)  :: b
      real(knd),intent(in)  :: a
      integer :: j

      !$omp parallel do
      do j=1,size(to,2)
        to(:,j) = b * to(:,j) + a
      end do
      !$omp end parallel do
    end subroutine

    subroutine multiply_and_add_scalar_3D(to,b,a)
      ! to = b * to + a
      real(knd),contiguous,intent(inout) :: to(:,:,:)
      real(knd),intent(in)  :: b
      real(knd),intent(in)  :: a
      integer :: k

      !$omp parallel do
      do k=1,size(to,3)
        to(:,:,k) = b * to(:,:,k) + a
      end do
      !$omp end parallel do
    end subroutine


    subroutine reciprocal_1D(to,a)
      ! to = a / to
      real(knd),contiguous,intent(inout) :: to(:)
      real(knd),intent(in)  :: a
      integer :: i

      !$omp parallel do
      do i=1,size(to)
        to(i) = a / to(i)
      end do
      !$omp end parallel do
    end subroutine

    subroutine reciprocal_2D(to,a)
      ! to = a / to
      real(knd),contiguous,intent(inout) :: to(:,:)
      real(knd),intent(in)  :: a
      integer :: j

      !$omp parallel do
      do j=1,size(to,2)
        to(:,j) = a / to(:,j)
      end do
      !$omp end parallel do
    end subroutine

    subroutine reciprocal_3D(to,a)
      ! to = a / to
      real(knd),contiguous,intent(inout) :: to(:,:,:)
      real(knd),intent(in)  :: a
      integer :: k

      !$omp parallel do
      do k=1,size(to,3)
        to(:,:,k) = a / to(:,:,k)
      end do
      !$omp end parallel do
    end subroutine

    subroutine add_element_int(a,e)
      integer,allocatable,intent(inout) :: a(:)
      integer,intent(in) :: e
      integer,allocatable :: tmp(:)

      if (.not.allocated(a)) then
        a = [e]
      else
        call move_alloc(a,tmp)
        allocate(a(size(tmp)+1))
        a(1:size(tmp)) = tmp
        a(size(tmp)+1) = e
      end if
    end subroutine
    
    pure function avg_real_1D(a) result(res)
      real(knd) :: res
      real(knd),intent(in) :: a(:)
      
      res = sum(a)/size(a)
    end function
    
    pure function avg_real_2D(a) result(res)
      real(knd) :: res
      real(knd),intent(in) :: a(:,:)
      
      res = sum(a)/size(a)
    end function
    
    pure function avg_real_3D(a) result(res)
      real(knd) :: res
      real(knd),intent(in) :: a(:,:,:)
      
      res = sum(a)/size(a)
    end function
    
    pure function avg_int_1D(a) result(res)
      integer :: res
      integer,intent(in) :: a(:)
      
      res = sum(a)/size(a)
    end function
    
    pure function avg_int_2D(a) result(res)
      integer :: res
      integer,intent(in) :: a(:,:)
      
      res = sum(a)/size(a)
    end function
    
    pure function avg_int_3D(a) result(res)
      integer :: res
      integer,intent(in) :: a(:,:,:)
      
      res = sum(a)/size(a)
    end function
    
    pure function dist(a,b) result(res)   
      real(knd) :: res
      real(knd),intent(in) :: a(3), b(3)

      res = norm2(b - a)
    end function
      
    function cross_product(a,b) result(axb)
      real(knd),dimension(3) :: axb
      real(knd),dimension(3),intent(in) :: a
      real(knd),dimension(3),intent(in) :: b 
       
      axb(1) = a(2)*b(3) - a(3)*b(2)
      axb(2) = a(3)*b(1) - a(1)*b(3)
      axb(3) = a(1)*b(2) - a(2)*b(1)
    end function
    
end module ArrayUtilities

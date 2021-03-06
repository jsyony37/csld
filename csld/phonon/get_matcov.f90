module get_matcov

  implicit none
  public pmatcov
  public bubblesort

contains

!----------------------------------------------
!
!       Compute Quantum Covariance Matrix
!       Yi Xia  yxia@anl.gov
!       ifort -o get_matcov-2000 get_matcov.f90 -mkl
!       f2py  -lmkl_rt -c -m get_matcov  get_matcov.f90
!----------------------------------------------
subroutine pmatcov(temperature,natom,mass,path,feng,matcov)
implicit none

! define parameters----------------------------
integer natom, i, j, ia, ib, ndim, l, ifreq, ifreq0, ja, jb
real*8  nphonon, temperature, mass(:), freqtmp, ratio
real*8, intent(out) :: matcov(3*natom,3*natom)
real*8, intent(out) :: feng
character(len=100) :: path
real*8, dimension(:,:), allocatable :: ifcs, ifcs2
real*8, dimension(:,:), allocatable :: dyn, dyn2
real*8, dimension(:), allocatable :: eig, freq, eig2, freq2
logical :: file_exists

complex(kind=8), allocatable :: work(:)
real(kind=8), allocatable :: rwork(:) 
integer, allocatable :: indexfreq(:), indexfreq2(:)
integer(kind=4) :: nwork=1, info=0

!f2py intent(in) :: temperature
!f2py intent(in) :: path

! read files-----------------------------------
!temperature = 2000 ! in K

open(10, file=trim(path)//'FORCE_CONSTANTS_2ND', status='old', form='formatted')
read(10, *) natom
!print *, "Number of atoms: "
!print *, natom
ndim=3*natom
allocate(rwork(max(1,9*natom-2)))
allocate(work(nwork))
allocate(ifcs(3*natom, 3*natom), ifcs2(3*natom, 3*natom))
allocate(dyn(3*natom, 3*natom), dyn2(3*natom, 3*natom))
!allocate(matcov(3*natom, 3*natom))
allocate(eig(3*natom), eig2(3*natom))
allocate(freq(3*natom), freq2(3*natom))
do i=1, natom
   do j=1, natom
      read(10, *) ia, ib
      read(10, *) ifcs((i-1)*3+1, (j-1)*3+1:j*3)
      read(10, *) ifcs((i-1)*3+2, (j-1)*3+1:j*3)
      read(10, *) ifcs((i-1)*3+3, (j-1)*3+1:j*3)
   end do
end do
close(10)


INQUIRE(FILE="FORCE_CONSTANTS_2ND_OLD", EXIST=file_exists)
if (file_exists) then
   open(11, file=trim(path)//'FORCE_CONSTANTS_2ND_OLD', status='old', form='formatted')
else
   open(11, file=trim(path)//'FORCE_CONSTANTS_2ND', status='old', form='formatted')
end if
read(11, *) natom
do i=1, natom
   do j=1, natom
      read(11, *) ia, ib
      read(11, *) ifcs2((i-1)*3+1, (j-1)*3+1:j*3)
      read(11, *) ifcs2((i-1)*3+2, (j-1)*3+1:j*3)
      read(11, *) ifcs2((i-1)*3+3, (j-1)*3+1:j*3)
   end do
end do
close(11)

!open(10, file=trim(path)//'masses', status='old', form='formatted')
!do i=1, natom
!   read(10, *) mass(i)
!end do
!close(10)
!print *, "Atomic Masses: "
!print *, mass(:)


! processing----------------------------------
! mass term
do i=1, natom
   do j=1, natom
      dyn((i-1)*3+1:i*3, (j-1)*3+1:j*3)=ifcs((i-1)*3+1:i*3, (j-1)*3+1:j*3)/sqrt(mass(i)*mass(j))
      dyn2((i-1)*3+1:i*3, (j-1)*3+1:j*3)=ifcs2((i-1)*3+1:i*3, (j-1)*3+1:j*3)/sqrt(mass(i)*mass(j))
   end do
end do
! conjugate and transpose to Hermitian
dyn=(dyn+transpose(dyn))/2.0
dyn2=(dyn2+transpose(dyn2))/2.0

!print *, "Check DYM: "
!do i=1, 1
!   do j=1, 2
!      print '(6f12.3)', dyn((i-1)*3+1, (j-1)*3+1:j*3)
!      print '(6f12.3)', dyn((i-1)*3+2, (j-1)*3+1:j*3)
!      print '(6f12.3)', dyn((i-1)*3+3, (j-1)*3+1:j*3)
!      print *
!   end do
!end do
print *, "Dynamical matrix assebled!"


!call heev(dyn, eig)

! diagonalize
!call zheev('V', 'U', ndim, dyn, ndim, eig, work, -1, rwork, info)
!print *, "Second step"
!if(real(work(1)).gt.nwork) then
!   nwork=nint(2*real(work(1)))
!   deallocate(work)
!   allocate(work(nwork))
!end if
!print *, "nwork"
!print *, nwork
!print *, "info"
!print *, info
! call zheev('V', 'U', ndim, dyn, ndim, eig, work, nwork, rwork, info)
!---------------------------------------------------------!
!Calls the LAPACK diagonalization subroutine DSYEV        !
!input:  a(n,n) = real symmetric matrix to be diagonalized!
!            n  = size of a                               !
!output: a(n,n) = orthonormal eigenvectors of a           !
!        eig(n) = eigenvalues of a in ascending order     !
!---------------------------------------------------------!

! assembling dynamical matrix
l=ndim*(3+ndim/2)
deallocate(work)
allocate(work(ndim*(3+ndim/2)))
call dsyev('V','U',ndim,dyn,ndim,eig,work,l,info)
call dsyev('V','U',ndim,dyn2,ndim,eig2,work,l,info)
! change to frequency in THz 98.1761/2.0/3.1416
do i=1, 3*natom
   if (eig(i) < 0) then
      freq(i)=-1*sqrt(-1*eig(i))*98.1761/2.0/3.1416
   else
      freq(i)=sqrt(eig(i))*98.1761/2.0/3.1416
   end if
   if (eig2(i) < 0) then
      freq2(i)=-1*sqrt(-1*eig2(i))*98.1761/2.0/3.1416
   else
      freq2(i)=sqrt(eig2(i))*98.1761/2.0/3.1416
   end if
end do
print *, "Frequencies (THz): "
print '(8f12.8)', freq(:)
!print *, "Eigenvevtor normalization: "
!print *, dot_product(dyn(:,1), dyn(:,1))

! sort frequency (with imaginary frequency)
allocate(indexfreq(3*natom), indexfreq2(3*natom))
indexfreq=0
do i=1, 3*natom
   indexfreq(i)=i
   indexfreq2(i)=i
   !take absolute value of frequency
   freq(i)=abs((freq(i)))
   freq2(i)=abs((freq2(i)))
end do
call bubblesort(ndim, freq, indexfreq)
call bubblesort(ndim, freq2, indexfreq2)
!print *, "1. Sorted Frequencies (THz): "
!print '(8f12.8)', freq(:)
!print *, "1. Index reference: "
!print '(12i6)', indexfreq(:)

!print *, "2. Sorted Frequencies (THz): "
!print '(8f12.8)', freq2(:)
!print *, "2. Index reference: "
!print '(12i6)', indexfreq2(:)

! construct correlation matrix
feng=0.d0
matcov=0.d0
ratio=0.d0
do ifreq0=1, ndim
   ifreq=indexfreq(ifreq0)
   freqtmp=(freq(ifreq0)*0.5+freq2(ifreq0)*0.5)
   if (freqtmp .gt. 1.0d-3 ) then
      ratio=freqtmp*4.13567/(0.086173*temperature)
      !feng=feng+0.5*freqtmp*4.13567-( 0.086173*log(1.0-exp(-1*ratio))+0.086173*ratio/(exp(ratio)-1.0) )*temperature
      feng=feng+0.5*freqtmp*4.13567+0.086173*temperature*log(1.0-exp(-1.0*ratio))  ! in meV
   end if
!   print '("Index of freq: ", i6)', ifreq0
!   print '("Original freq: ", f12.8)', freqtmp
   if (freqtmp < 1.0d-3) then
      freqtmp=100000.0
!      print *, "Change frequency to very large values 100000 THZ"
!      print *, freqtmp
   end if
   !print '("Phonon freq: ", f12.8)', freqtmp
   nphonon=(1.0+2.0/(exp(47.9924*freqtmp/temperature)-1.0))
!   print '( "Number of phonons: ", f12.6)', nphonon
   do ia=1, natom
      do ib=1, natom
         do ja=1, 3
            do jb=1, 3
               matcov((ia-1)*3+ja, (ib-1)*3+jb) = matcov((ia-1)*3+ja, (ib-1)*3+jb)+ &
                    0.5*6.35039/2/3.1416/freqtmp*nphonon / &
                    sqrt(mass(ia)*mass(ib)) * &
                    dyn((ia-1)*3+ja,ifreq)*dyn((ib-1)*3+jb,ifreq)
!                    dyn(ifreq, (ia-1)*3+ja)*dyn(ifreq, (ib-1)*3+jb)
            end do
         end do
      end do
   end do
end do

print *, "Check matcov: "
do i=1, 1
   do j=1, 2
      print '(3f12.9)', matcov((i-1)*3+1, (j-1)*3+1:j*3)
      print '(3f12.9)', matcov((i-1)*3+2, (j-1)*3+1:j*3)
      print '(3f12.9)', matcov((i-1)*3+3, (j-1)*3+1:j*3)
      print *
   end do
end do

open(33, file=trim(path)//'matcov.dat', form='formatted', status='unknown')
write(33, *) ndim
do i=1, ndim
write(33, '(100000E20.10)') matcov(i, :)
end do
close(33)

feng = feng/natom
open(33, file=trim(path)//'free_eng.dat', form='formatted', status='unknown') 
write(33, *) "Free energy per atom in meV"
write(33, '(f12.6)') feng
close(33)

end subroutine


! -------------------------------------------------------------------
! bubble sort algorithm
subroutine bubblesort(len, freq, index)
  integer, intent(in) :: len
  real*8, intent(inout) :: freq(len)
  integer, intent(inout) :: index(len)
  
  real*8 tmp, tmp0
  integer i, j, ind, ind0

!  print *, "Before sorted Frequencies (THz): "
!  print '(3f12.8)', freq(:)
!  print *, "Starting bubble sorting:"
  do i=len, 2, -1
     do j=2, i
        if ( freq(j-1).gt.freq(j) ) then
           tmp=freq(j)
           tmp0=freq(j-1)
           freq(j)=tmp0
           freq(j-1)=tmp
           ind=index(j)
           ind0=index(j-1)
           index(j)=ind0
           index(j-1)=ind
        end if
     end do
  end do
!  print *, "After sorted Frequencies (THz): " 
!  print '(3f12.8)', freq(:)
end subroutine bubblesort

end module

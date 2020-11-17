module gruneisen

  implicit none
  public phexp
  public read3fc
  public mode_grun
  public total_grun
  
contains

  ! exp(iunit*x). Sometimes it is much faster than the general
  ! complex exponential. 
  function phexp(x)
    implicit none

    real(kind=8),intent(in) :: x
    complex(kind=8) :: phexp

    phexp=cmplx(cos(x),sin(x),kind=8)
  end function phexp


    ! Read FORCE_CONSTANTS_3RD.
  subroutine read3fc(lattvec,Ntri,Phi,R_j,R_k,Index_i,Index_j,Index_k)
    implicit none
    real(kind=8), intent(in) :: lattvec(3,3)
    integer(kind=4), intent(in) :: Ntri
    integer(kind=4), intent(out) :: Index_i(Ntri),Index_j(Ntri),Index_k(Ntri)
    real(kind=8), intent(out) :: Phi(3,3,3,Ntri),R_j(3,Ntri),R_k(3,Ntri)

    real(kind=8) :: tmp(3,3)
    integer(kind=4) :: ii,jj,ll,mm,nn,ltem,mtem,ntem,info,P(3)

    ! The file is in a simple sparse format, described in detail in
    ! the user documentation. See Doc/ShengBTE.pdf.
    open(1,file='FORCE_CONSTANTS_3RD',status="old")
    read(1,*)
    do ii=1,Ntri
       read(1,*) jj
       read(1,*) R_j(:,ii)
       read(1,*) R_k(:,ii)
       read(1,*) Index_i(ii),Index_j(ii),Index_k(ii)
       do ll=1,3
          do mm=1,3
             do nn=1,3
                read(1,*) ltem,mtem,ntem,Phi(ll,mm,nn,ii)
             end do
          end do
       end do
    end do
    close(1)
    ! Each vector is rounded to the nearest lattice vector.
    tmp=lattvec
    call dgesv(3,Ntri,tmp,3,P,R_j,3,info)
    R_j=matmul(lattvec,anint(R_j))
    tmp=lattvec
    call dgesv(3,Ntri,tmp,3,P,R_k,3,info)
    R_k=matmul(lattvec,anint(R_k))
  end subroutine read3fc

  
  ! Subroutine to compute the mode Gruneisen parameters.
  subroutine mode_grun(omega,eigenvect,Phi,R_j,R_k,Index_i,Index_j,Index_k,cartesian,masses,kspace,&
       nbands,nptk,natoms,Ntri,grun)
    implicit none

    integer(kind=4), intent(in) :: nbands,nptk,natoms,Ntri
    integer(kind=4), intent(in) :: Index_i(Ntri),Index_j(Ntri),Index_k(Ntri)
    real(kind=8), intent(in) :: omega(nptk,nbands),masses(natoms),cartesian(3,natoms)
    real(kind=8), intent(in) :: Phi(3,3,3,Ntri),R_j(3,Ntri),R_k(3,Ntri)
    complex(kind=8), intent(in) :: eigenvect(nptk,nbands,nbands)
    real(kind=8), intent(out) :: grun(nptk,nbands)

!    real(kind=8),parameter :: unitfactor=9.6472d4 ! From nm*eV/(amu*A^3*THz^2) to SI=1.
    real(kind=8), parameter :: unitfactor=9.6472d3 ! From A*eV/(amu*A^3*THz^2) to SI=1.
    real(kind=8), parameter :: pi=3.141592d0
    
    integer(kind=4) :: ik,ii,jj,kk,iband,itri,ialpha,ibeta!,ngrid(3)
    real(kind=8) :: kspace(nptk,3)
    complex(kind=8) :: factor1,factor2,factor3,g(nbands)

    print *, "# OF KPOINTS : ",nptk
    print *, "KSPACE DIM : ",shape(kspace)
    print *, "CARTESIAN :"
    print *, cartesian
    print *, "MASSES :"
    print *, masses
    print *, "OMEGA at GAMMA:"
    print *, omega(1,:)
    print *, "EIGENVECTOR at GAMMA:"
    write(*, '(18F0.4)') real(eigenvect(1,:,:))
    open(1, file = 'KSPACE', status = 'replace')
    write(1, '(3F0.4)') transpose(kspace)
    close(1)    
    
    ! Calculate mode Gruneisen parameters
    do ik=1,nptk
       g=0.0
       do iband=1,nbands
          do itri=1,Ntri
             factor1=phexp(dot_product(kspace(ik,:),R_j(:,itri)))/&
!                  sqrt(masses(types(Index_i(itri)))*masses(types(Index_j(itri))))
                  sqrt(masses(Index_i(itri))*masses(Index_j(itri)))
             do ialpha=1,3
                factor2=factor1*conjg(eigenvect(ik,iband,3*(Index_i(itri)-1)+ialpha))
                do ibeta=1,3
                   factor3=factor2*eigenvect(ik,iband,3*(Index_j(itri)-1)+ibeta)
                   g(iband)=g(iband)+factor3*dot_product( Phi(ialpha,ibeta,:,itri) , cartesian(:,Index_k(itri))+R_k(:,itri) )
                end do
             end do
          end do
          if (abs(omega(ik,iband)) < 1d-03) then
             grun(ik,iband)=0.d0
          else
             g(iband)=-unitfactor*g(iband)/6.d00/omega(ik,iband)**2
             grun(ik,iband)=real(g(iband))
          endif
       end do
    end do

  end subroutine mode_grun

  
  ! Obtain the total Gruneisen parameter as a weighted sum over modes.
  function total_grun(omega,grun,T,nptk,nbands)
    implicit none

    integer(kind=4), intent(in) :: nptk,nbands
    real(kind=8), intent(in) :: omega(nptk,nbands)
    real(kind=8), intent(in) :: grun(nptk,nbands),T
    real(kind=8), parameter :: hbar=1.05457172647d-22 ! J/THz
    real(kind=8), parameter :: kB=1.380648813d-23 ! J/K
    real(kind=8) :: weight,dBE,x
    integer(kind=4) :: ik,iband
    real(kind=8) :: total_grun

    total_grun=0.
    weight=0.
    do iband=1,nbands
       do ik=1,nptk
          x=hbar*omega(ik,iband)/(2.*kB*T)
          if(x.gt.1e-6) then
             dBE=(x/sinh(x))**2.
             weight=weight+dBE
             total_grun=total_grun+dBE*grun(ik,iband)
          end if
       end do
    end do
    total_grun=total_grun/weight
  end function total_grun

end module

      SUBROUTINE DOSTET(ENER,IDIME,NBAND,IDEF,NTET,XE,NE,Y,Z)
C     COMPUTE A DENSITY OF STATES USING THE TETRAHEDRONS METHOD.
C     XE CONTAINS THE ENERGIES, Y AND Z RETURN THE RELATED DENSITY OF
C     STATES AND THE INTEGRATED DENSITY OF STATES, RESPECTIVELY.
      IMPLICIT REAL*8(A-H,O-Z)
      REAL*4 AVOL
      DIMENSION ENER(IDIME,1),XE(1),Y(1),Z(1),IDEF(5,1),C(4)
      EQUIVALENCE (IVOL,AVOL),(C(1),E1),(C(2),E2),(C(3),E3),(C(4),E4)
      DATA EPS/1.0D-05/
      DO 6 IX=1,NE
      Y(IX)=0.D0
    6 Z(IX)=0.D0
C
C     LOOP OVER THE TETRAHEDRONS
      DO 9 ITET=1,NTET
C
      IA=IDEF(1,ITET)
      IB=IDEF(2,ITET)
      IC=IDEF(3,ITET)
      ID=IDEF(4,ITET)
      IVOL=IDEF(5,ITET)
C
C     LOOP OVER THE BANDS
      DO 9 NB=1,NBAND
C
C *** DEFINE E1, E2, E3, E4, AS THE CORNER ENERGIES ORDERED BY
C *** DECREASING SIZE
      C(1)=ENER(NB,IA)
      C(2)=ENER(NB,IB)
      C(3)=ENER(NB,IC)
      C(4)=ENER(NB,ID)
      DO 2 I=1,4
      CC=C(I)
      J=I
    1 J=J+1
      IF(J.GT.4) GOTO 2
      IF(CC.GE.C(J)) GOTO 1
      C(I)=C(J)
      C(J)=CC
      CC=C(I)
      GOTO 1
    2 CONTINUE
      UNITE=1.0D0
      IF(E1.GT.E4) UNITE=E1-E4
      E12=(E1-E2)/UNITE
      E13=(E1-E3)/UNITE
      E14=(E1-E4)/UNITE
      E23=(E2-E3)/UNITE
      E24=(E2-E4)/UNITE
      E34=(E3-E4)/UNITE
      FACY=3.D0*DBLE(AVOL)/UNITE
      DO 9 IX=1,NE
      E=XE(IX)
      SURFAC=0.D0
      VOLUME=1.D0
      IF(E.GT.E1) GOTO 8
      VOLUME=0.D0
      IF(E.LT.E4) GOTO 8
      EE1=(E-E1)/UNITE
      IF(DABS(EE1).LT.EPS) EE1=0.D0
      EE2=(E-E2)/UNITE
      IF(DABS(EE2).LT.EPS) EE2=0.D0
      EE3=(E-E3)/UNITE
      IF(DABS(EE3).LT.EPS) EE3=0.D0
      EE4=(E-E4)/UNITE
      IF(DABS(EE4).LT.EPS) EE4=0.D0
      IF(E.GT.E3) GOTO 5
C *** E4.LE.E.AND.E.LE.E3
      IF(E4.EQ.E3) GOTO 3
      SURFAC=(EE4/E34)*(EE4/E24)
      VOLUME=SURFAC*EE4
      GOTO 8
    3 IF(E3.LT.E2) GOTO 8
      IF(E2.EQ.E1) GOTO 4
      SURFAC=1.D0/E12
      GOTO 8
    4 SURFAC=1.0D+15
      VOLUME=0.5D0
      GOTO 8
    5 IF(E.GT.E2) GOTO 7
C *** E3.LT.E.AND.E.LE.E2
      SURFAC=-(EE3*EE2/E23+EE4*EE1)/E13/E24
      VOLUME=(0.5D0*EE3*(2.D0*E13*E34+E13*EE4-E34*EE1-2.D0*EE1*EE4+
     +                 EE3*(EE3-3.D0*EE2)/E23)/E13+E34*E34)/E24
      GOTO 8
C *** E2.LT.E.AND.E.LE.E1
    7 SURFAC=(EE1/E12)*(EE1/E13)
      VOLUME=1.D0+SURFAC*EE1
    8 Y(IX)=Y(IX)+FACY*SURFAC
      Z(IX)=Z(IX)+DBLE(AVOL)*VOLUME
    9 CONTINUE
      RETURN
      END

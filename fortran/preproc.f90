module preproc
    use ibstructs
    implicit none

    private t_acc

    real(dp), parameter :: ig_f=240.0e3_dp, ig_t=2427.0_dp, ig_theta = 0.22_dp
    contains
        subroutine compute_igniter(ign, ibp, powds)
            ! Расчет воспламенителя
            type(igniter), intent(out) :: ign
            type(ibproblem), intent(in) :: ibp
            type(powder), dimension(:), intent(in) :: powds

            real(dp) :: igniter_mass
            real(dp) :: ro_om_sum
            
            integer :: i, n_powd

            ro_om_sum = 0.0_dp
            n_powd = size(powds)

            if (ibp%ig_mass > 0.0_dp) then
                igniter_mass = ibp%ig_mass
            else
                do i = 1, n_powd
                    ro_om_sum = ro_om_sum + powds(i)%om/powds(i)%ro
                end do
                igniter_mass = ibp%pv*(ibp%artsystem%w0 - ro_om_sum)/ig_f
            end if

            ign%fs = igniter_mass*ig_f
            ign%num = ign%fs/ig_t
            ign%denum = ign%num/ig_theta

        end subroutine compute_igniter

        subroutine temp_account(charge, t)
            type(powder), dimension(:), intent(inout) :: charge
            real(dp), intent(in) :: t

            integer :: i, n_powd
            
            n_powd = size(charge)

            do i = 1, n_powd
               call t_acc(charge(i), t)
            end do
        end subroutine

        subroutine t_acc(powd, t)
            type(powder), intent(inout) :: powd
            real(dp), intent(in) :: t

            powd%jk = powd%jk*(1 - powd%gamma_jk*(t - 15.0_dp))
            powd%f_powd = powd%f_powd*(1 + powd%gamma_f*(t - 15.0_dp))

        end subroutine t_acc
end module preproc
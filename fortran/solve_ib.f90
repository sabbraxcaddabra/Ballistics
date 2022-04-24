module solve_ib
    use preproc
    use iso_c_binding
    implicit none

    private compute_om_sum, theta_av, compute_psi, pressure, int_bal_rs, k50
    real(c_double), parameter :: k50 = 50e6_dp ** 0.25_dp

    contains

        real(c_double) function compute_om_sum(charge) result(om_sum)
            type(powder), dimension(:), intent(in) :: charge

            integer(c_int) :: i, n_powd
            
            om_sum = 0.0_dp
            n_powd = size(charge)

            do i = 1, n_powd
                om_sum = om_sum + charge(i)%om
            end do
        end function compute_om_sum

        real(c_double) function theta_av(psis, ign, charge)
        ! Расчет осредненного(average) параметра расширения
            real(c_double), dimension(:), intent(in) :: psis
            type(igniter), intent(in) :: ign
            type(powder), dimension(:), intent(in) :: charge

            real(c_double) :: num, denum
            integer(c_int) :: i, n_powd

            n_powd = size(charge)
            num = ign%num
            denum = ign%denum

            do i = 1, n_powd
                num = num + charge(i)%f_powd * charge(i)%om * psis(i)/charge(i)%ti
                denum = denum + charge(i)%f_powd * charge(i)%om * psis(i)/(charge(i)%ti * charge(i)%theta)
            end do

            if (denum > 0.0_dp) then
                theta_av = num/denum
            else
                theta_av = 0.4_dp
            end if

        end function theta_av

        subroutine compute_psi(psis, z, charge)
            ! Расчет относительной массы сгоревшего пороха с записью в массив psis
            real(c_double), dimension(:), intent(inout) :: psis
            real(c_double), dimension(:), intent(in) :: z
            type(powder), dimension(:), intent(in) :: charge

            real(c_double) :: z1, psi_s 
            integer(c_int) :: i, n_powd

            n_powd = size(charge)

            do i = 1, n_powd
                if (z(i) < 1.0_dp) then
                    psis(i) = charge(i)%kappa1*z(i)*(1.0_dp + charge(i)%lambda1*z(i) + charge(i)%mu1*z(i)**2)
                elseif (z(i) < charge(i)%zk) then
                    z1 = z(i) - 1.0_dp
                    psi_s = charge(i)%kappa1*(1.0_dp + charge(i)%lambda1 + charge(i)%mu1)
                    psis(i) = psi_s + charge(i)%kappa2*z1*(1.0_dp + charge(i)%lambda2*z1 + charge(i)%mu2*z1**2)
                else
                    psis(i) = 1.0_dp
                end if

                if (psis(i) > 1.0_dp) then
                    psis(i) = 1.0_dp
                end if
                
            end do
        end subroutine compute_psi

        subroutine pressure(p, y, ign, barl, l_chi, om_sum, psis, charge, err_state)
            real(c_double), dimension(:), intent(inout) :: p(3)
            real(c_double), dimension(:), intent(in) :: y
            type(igniter), intent(in) :: ign
            type(barrel), intent(in) :: barl
            real(c_double), intent(in) :: l_chi
            real(c_double), intent(in) :: om_sum
            real(c_double), dimension(:), intent(in) :: psis
            type(powder), dimension(:), intent(in) :: charge
            integer(c_int), intent(out) :: err_state

            real(c_double) :: theta, fs, w
            integer(c_int) i, n_powd

            n_powd = size(charge)

            theta = theta_av(psis, ign, charge)
            fs = ign%fs
            w = barl%w0

            do i = 1, n_powd
                fs = fs + charge(i)%f_powd * charge(i)%om * psis(i)
                w = w - charge(i)%om*((1.0_dp - psis(i))/charge(i)%ro + psis(i)*charge(i)%alpha)
            end do

            fs = fs - theta*y(1)**2 * (0.5_dp * barl%q + l_chi*om_sum/6.0_dp)
            w = w + y(2)*barl%s
            if (w < 0.0_dp) then
                err_state = 1
                return
            end if

            p(1) = 1e5_dp + fs/w
            if (y(1) > 0.0_dp) then
                p(2) = p(1)/(1.0_dp + (1.0_dp/3.0_dp)*(l_chi*om_sum)/barl%q)
                p(3) = p(2)*(1.0_dp + 0.5_dp*l_chi*om_sum/barl%q)
            else
                p(2) = p(1)
                p(3) = p(1)
            end if

        end subroutine pressure

        subroutine int_bal_rs(dy, y, psis, p, p0, om_sum, ign, barl, charge, err_state)
            real(c_double), dimension(:), intent(out) :: dy
            real(c_double), dimension(:), intent(in) :: y
            real(c_double), dimension(:), intent(inout) :: psis
            real(c_double), dimension(:), intent(inout) :: p
            real(c_double), intent(in) :: p0
            real(c_double), intent(in) :: om_sum
            type(igniter), intent(in) :: ign
            type(barrel), intent(in) :: barl
            type(powder), dimension(:), intent(in) :: charge
            integer(c_int), intent(out) :: err_state

            real(c_double) :: l_chi
            integer(c_int) i, n_powd
            n_powd = size(charge)
            
            l_chi = (y(2) + barl%l_k)/(y(2) + barl%l0)

            call compute_psi(psis, y(3:), charge)
            call pressure(p, y, ign, barl, l_chi, om_sum, psis, charge, err_state)

            if (err_state == 1) then
                return
            end if

            if ((y(1) > 0.0) .or. (p(1) > p0)) then
                dy(1) = p(2) * barl%s/barl%q
                dy(2) = y(1)
            else
                dy(1) = 0.0_dp 
                dy(2) = 0.0_dp
            end if

            do i = 1, n_powd
                if (p(1) < 50e6_dp) then
                    dy(2 + i) = ((k50*p(1)**0.75_dp)/charge(i)%jk)
                else
                    dy(2 + i) = (p(1)/charge(i)%jk)
                end if
            end do

        end subroutine int_bal_rs

        subroutine count_ib(ibp, n_powd, charge, tstep, tend) bind(c, name='count_ib')

            type(ibproblem), intent(inout) :: ibp
            integer(c_int), intent(in), value :: n_powd
            type(powder), dimension(:), intent(inout) :: charge(n_powd)
            real(c_double), intent(in), value :: tstep, tend

            real(c_double), dimension(:, :) :: dy(2 + n_powd, 4)
            real(c_double), dimension(:) :: y(2 + n_powd)
            real(c_double), dimension(:) :: psis(n_powd)
            real(c_double), dimension(:) :: p(3)
            logical(c_bool) :: burned
            real(c_double) :: t0, om_sum, p0, psi_sum
            integer(c_int) :: i, err_state

            type(barrel) :: barl
            type(igniter) :: ign

            barl = ibp%artsystem
            barl%q = barl%q * barl%kf

            t0 = 0.0_dp
            dy = 0.0_dp
            y = 0.0_dp
            psis = 0.0_dp
            p = 1e5_dp

            burned = .false.
            err_state = 0
            om_sum = compute_om_sum(charge)
            p0 = ibp%p0

            call compute_igniter(ign, ibp, charge)

            call temp_account(charge, ibp%t0)

            do while (y(2) < barl%l_d)
                if (all(psis >= 1) .and. .not. burned) then
                    ibp%eta_k = y(2)/barl%l_d
                    burned = .true.
                end if

                call int_bal_rs(dy(:, 1), y, psis, p, p0, om_sum, ign, barl, charge, err_state)
                ibp%p_av_max = max(ibp%p_av_max, p(1))
                ibp%p_sn_max = max(ibp%p_sn_max, p(2))
                ibp%p_kn_max = max(ibp%p_kn_max, p(3))
                call int_bal_rs(dy(:, 2), y + 0.5_dp*tstep*dy(:, 1), psis, p, p0, om_sum, ign, barl, charge, err_state)
                call int_bal_rs(dy(:, 3), y + 0.5_dp*tstep*dy(:, 2), psis, p, p0, om_sum, ign, barl, charge, err_state)
                call int_bal_rs(dy(:, 4), y + tstep*dy(:, 3), psis, p, p0, om_sum, ign, barl, charge, err_state)

                if (err_state == 1) then
                    ibp%status = 1
                    return
                end if

                t0 = t0 + tstep
                y = y + tstep*(dy(:, 1) + 2*dy(:, 2) + 2*dy(:, 3) + dy(:, 4))/6
                
                if (t0 > tend) then
                    ibp%status = 2
                    return
                end if
            end do

            psi_sum = 0.0_dp

            call compute_psi(psis, y(3:), charge)

            do i = 1, n_powd
                psi_sum = psi_sum + psis(i) * charge(i)%om
            end do

            ibp%v0 = y(1)
            ibp%psi_sum = psi_sum/om_sum

        end subroutine count_ib

        subroutine dense_count_ib(ibp, n_powd, charge, tstep, tend,&
                                  n_tsteps, y_array, pressure_array) bind(c, name='dense_count_ib')

            type(ibproblem), intent(inout) :: ibp
            integer(c_int), intent(in), value :: n_powd
            integer(c_int), intent(inout) :: n_tsteps
            real(c_double), intent(inout) :: y_array(2 + n_powd, n_tsteps), pressure_array(3, n_tsteps)
            type(powder), dimension(:), intent(inout) :: charge(n_powd)
            real(c_double), intent(in), value :: tstep, tend

            real(c_double), dimension(:, :) :: dy(2 + n_powd, 4)
            real(c_double), dimension(:) :: psis(n_powd)
            real(c_double), dimension(:, :) :: p(3, 4)
            logical(c_bool) :: burned
            real(c_double) :: t0, om_sum, p0, psi_sum
            integer(c_int) :: i, err_state

            type(barrel) :: barl
            type(igniter) :: ign

            barl = ibp%artsystem
            barl%q = barl%q * barl%kf

            t0 = 0.0_dp
            y_array(:, 1) = 0.0_dp
            dy = 0.0_dp
            psis = 0.0_dp
            p = 1e5_dp

            burned = .false.
            err_state = 0
            om_sum = compute_om_sum(charge)
            p0 = ibp%p0

            call compute_igniter(ign, ibp, charge)

            call temp_account(charge, ibp%t0)

            do i = 2, n_tsteps
                if (all(psis >= 1) .and. .not. burned) then
                    ibp%eta_k = y_array(2, i-1)/barl%l_d
                    burned = .true.
                end if

                call int_bal_rs(dy(:, 1), y_array(:, i-1), psis, p(:, 1), p0, om_sum, ign, barl, charge, err_state)
                pressure_array(:, i-1) = p(:, 1)
                call int_bal_rs(dy(:, 2), y_array(:, i-1) + 0.5_dp*tstep*dy(:, 1), psis, p(:, 2), p0, om_sum, ign, barl,&
                 charge, err_state)
                call int_bal_rs(dy(:, 3), y_array(:, i-1) + 0.5_dp*tstep*dy(:, 2), psis, p(:, 3), p0, om_sum, ign, barl,&
                 charge, err_state)
                call int_bal_rs(dy(:, 4), y_array(:, i-1) + tstep*dy(:, 2), psis, p(:, 4), p0, om_sum, ign, barl, charge,&
                 err_state)

                if (err_state == 1) then
                    ibp%status = 1
                    return
                end if

                ibp%p_av_max = max(ibp%p_av_max, p(1, 1))
                ibp%p_sn_max = max(ibp%p_sn_max, p(2, 1))
                ibp%p_kn_max = max(ibp%p_kn_max, p(3, 1))

                t0 = t0 + tstep
                y_array(:, i) = y_array(:, i-1) + tstep*(dy(:, 1) + 2*dy(:, 2) + 2*dy(:, 3) + dy(:, 4))/6

                call compute_psi(y_array(3:2+n_powd, i-1), y_array(3:2+n_powd, i-1), charge)

                if (t0 > tend) then
                    ibp%status = 2
                    return
                end if

                if (y_array(2, i) > barl%l_d) then
                    call int_bal_rs(dy(:, 1), y_array(:, i), psis, p(:, 1), p0, om_sum, ign, barl, charge, err_state)
                    y_array(3:2+n_powd, i) = psis
                    pressure_array(:, i) = p(:, 1)
                    n_tsteps = i
                    exit
                end if

                if (t0 > tend) then
                    ibp%status = 2
                    return
                end if
            end do

            psi_sum = 0.0_dp

            call compute_psi(psis, y_array(3:2+n_powd, n_tsteps), charge)

            do i = 1, n_powd
                psi_sum = psi_sum + psis(i) * charge(i)%om
            end do

            ibp%v0 = y_array(1, n_tsteps)
            ibp%psi_sum = psi_sum/om_sum

        end subroutine dense_count_ib

end module solve_ib